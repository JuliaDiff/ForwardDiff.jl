############################################################################
#
#  Reverse mode automatic differentiation
#
#  main entry point = reversediff(ex, out::Symbol, in as keyword-args)
#  returns (var allocation expressions, value + gradient expression, symbol of value in expression)
#
############################################################################

# where the derived functions are to be evaluated : the parent module of Autodiff
const parent_mod = Base.module_parent(current_module())

# naming conventions
const TEMP_NAME = "tmp"     # prefix of new variables
const DERIV_PREFIX = "d"   # prefix of gradient variables

##########  Parameterized type to ease AST exploration  ############
type ExH{H}
	head::Symbol
	args::Vector
	typ::Any
end
toExH(ex::Expr) = ExH{ex.head}(ex.head, ex.args, ex.typ)
toExpr(ex::ExH) = Expr(ex.head, ex.args...)

typealias ExEqual    ExH{:(=)}
typealias ExDColon   ExH{:(::)}
typealias ExPEqual   ExH{:(+=)}
typealias ExMEqual   ExH{:(-=)}
typealias ExTEqual   ExH{:(*=)}
typealias ExTrans    ExH{symbol("'")} 
typealias ExCall     ExH{:call}
typealias ExBlock	 ExH{:block}
typealias ExLine     ExH{:line}
typealias ExVcat     ExH{:vcat}
typealias ExRef      ExH{:ref}
typealias ExIf       ExH{:if}
typealias ExComp     ExH{:comparison}
typealias ExDot      ExH{:.}

## variable symbol sampling functions
getSymbols(ex::Any)    = Set{Symbol}()
getSymbols(ex::Symbol) = Set{Symbol}(ex)
getSymbols(ex::Array)  = mapreduce(getSymbols, union, ex)
getSymbols(ex::Expr)   = getSymbols(toExH(ex))
getSymbols(ex::ExH)    = mapreduce(getSymbols, union, ex.args)
getSymbols(ex::ExCall) = mapreduce(getSymbols, union, ex.args[2:end])  # skip function name
getSymbols(ex::ExRef)  = setdiff(mapreduce(getSymbols, union, ex.args), Set(:(:), symbol("end")) )# ':'' and 'end' do not count
getSymbols(ex::ExDot)  = Set{Symbol}(ex.args[1])  # return variable, not fields
getSymbols(ex::ExComp) = setdiff(mapreduce(getSymbols, union, ex.args), 
	Set(:(>), :(<), :(>=), :(<=), :(.>), :(.<), :(.<=), :(.>=), :(==)) )


## variable symbol subsitution functions
substSymbols(ex::Any, smap::Dict)     = ex
substSymbols(ex::Expr, smap::Dict)    = substSymbols(toExH(ex), smap::Dict)
substSymbols(ex::Vector, smap::Dict)  = map(e -> substSymbols(e, smap), ex)
substSymbols(ex::ExH, smap::Dict)     = Expr(ex.head, map(e -> substSymbols(e, smap), ex.args)...)
substSymbols(ex::ExCall, smap::Dict)  = Expr(:call, ex.args[1], map(e -> substSymbols(e, smap), ex.args[2:end])...)
substSymbols(ex::ExDot, smap::Dict)   = (ex = toExpr(ex) ; ex.args[1] = substSymbols(ex.args[1], smap) ; ex)
substSymbols(ex::Symbol, smap::Dict)  = get(smap, ex, ex)


## misc functions
dprefix(v::Union(Symbol, String, Char)) = symbol("$DERIV_PREFIX$v")
dprefix(v::Expr)                        = dprefix(toExH(v))
dprefix(v::ExRef)                       = Expr(:ref, dprefix(v.args[1]), v.args[2:end]...)
dprefix(v::ExDot)                       = Expr(:., dprefix(v.args[1]), v.args[2:end]...)

isSymbol(ex)   = isa(ex, Symbol)
isDot(ex)      = isa(ex, Expr) && ex.head == :.   && isa(ex.args[1], Symbol)
isRef(ex)      = isa(ex, Expr) && ex.head == :ref && isa(ex.args[1], Symbol)

## var name generator
let
	vcount = Dict()
	global newvar
	function newvar(radix::Union(String, Symbol)="")
		vcount[radix] = haskey(vcount, radix) ? vcount[radix]+1 : 1
		return symbol("$(radix)_$(vcount[radix])")
	end

	global resetvar
	function resetvar()
		vcount = Dict()
	end
end

######### structure for parsing model  ##############
type ParsingStruct
	bsize::Int                # length of beta, the parameter vector
	init::Vector 			  # initial values of input variables
	insyms::Vector{Symbol}    # input vars symbols
	outsym::Symbol            # output variable name (possibly renamed from initial out argument)
	source::Expr              # model source
	exprs::Vector{Expr}       # vector of assigments that make the model
	dexprs::Vector{Expr}      # vector of assigments that make the gradient

	ag::Dict                  # variable ancestors graph
	dg::Dict                  # variable descendants graph

	ParsingStruct() = new()   # uninitialized constructor
end

# find variables in dependency graph g
relations(v::Symbol, g)  = haskey(g, v) ? union( g[v], relations(g[v] ,g) ) : Set()
relations(vs::Vector, g) = union( map( s->relations(s,g) , vs)... )
relations(vs::Set, g)    = union( map( s->relations(s,g) , [vs...])... )

# active variables whose gradient need to be calculated
activeVars(m::ParsingStruct) = intersect(union(Set(m.outsym), relations(m.outsym, m.ag)), 
	                                     union(Set(m.insyms...), relations(m.insyms, m.dg)) )
# variables that are not defined in expression and are not input variables
external(m::ParsingStruct) = setdiff(union(values(m.ag)...), union(Set(keys(m.ag)...), Set(m.insyms...)))

##### now include parsing and derivation functions
include("deriv_rules.jl")
include("pass1.jl")
include("pass2.jl")



##################################################################
#  Main function definition
############################################################################

function reversediff(model::Expr, out::Symbol, skipgradient=false; init...)

   	length(init)>=1 || error("There should be at least one parameter specified, none found")

	m = ParsingStruct()
	m.source = model
	m.outsym = out
	m.insyms = map(sv->sv[1], init)
	m.init = map(sv->sv[2], init)

	unfold!(m)	

	m.ag, m.dg, subst, m.exprs = varGraph(m.exprs)

	m.outsym = get(subst, m.outsym, m.outsym) # update new name of outcome variable

	### controls 
	relations(m.outsym, m.ag) == Set() && error("outcome variable is not set or is constant")

	ui = setdiff(Set(m.insyms...), relations(m.outsym, m.ag))
	ui != Set() && error("some input variables ($(collect(ui))) do not influence outcome")

	# now generate 
	#  - 1 block for initialization statements : 'header'
	#  - 1 block for calculations : 'body'

	# identify external vars and add definitions x = Main.x
	header = [ :( local $v = $(Expr(:., :Main, Expr(:quote, v))) ) for v in external(m)]
	body = copy(m.exprs)

	if !skipgradient
		preCalculate(m)
		backwardSweep!(m)
		avars = activeVars(m) # active vars

		for v in avars 
			vh = vhint[v]
			dsym = dprefix(v)
			if isa(vh, Real)
				if v == m.outsym
					push!(body, :($dsym = 1.) )  # if final result backward propagation starts with 1.0 
				else
					push!(body, :($dsym = 0.) )
				end			
			# elseif 	isa(vh, LLAcc)
			# 	push!(body, :($(symbol("$dsym.1")) = 0.) )
			elseif 	isa(vh, Array{Float64})
				#  FIXME : inactivated to avoid having gradient vars rewritten on each call
				# push!(header, :( local $dsym = Array(Float64, $(Expr(:tuple,size(vh)...)))) )
				# push!(body, :( fill!($dsym, 0.) ) )
				push!(body, :( $dsym = zeros( $(Expr(:tuple,size(vh)...)))) )
			elseif 	isa(vh, Distribution)  #  TODO : find real equivalent vector size
				push!(body, :( $(symbol("$dsym.1")) = 0. ) )
				push!(body, :( $(symbol("$dsym.2")) = 0. ) )
			elseif 	isa(vh, Array) && isa(vh[1], Distribution)  #  TODO : find real equivalent vector size
				push!(header, :( local $(symbol("$dsym.1")) = Array(Float64, $(Expr(:tuple,size(vh)...)) ) ) )
				push!(header, :( local $(symbol("$dsym.2")) = Array(Float64, $(Expr(:tuple,size(vh)...)) ) ) )
				push!(body, :( fill!($(symbol("$dsym.1")), 0.) ) )
				push!(body, :( fill!($(symbol("$dsym.2")), 0.) ) )
			else
				warn("[diff] unknown type $(typeof(vh)), assuming associated gradient is Float64")
				push!(body, :($dsym = 0.) )
			end
		end

		body = vcat(body, m.dexprs) 
	end

	(Expr(:block, header...), Expr(:block, body...), m.outsym)
end




