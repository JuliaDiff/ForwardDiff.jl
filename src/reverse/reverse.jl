############################################################################
#  First attempt at decoupling automatic derivation from MCMC specific
#    code.
#
#  a unique entry point : diff(ex, out::Symbol, in as keyword-args)
#  returns an expression + var allocation expressions
#
############################################################################

	# where the derived functions are to be evaluated : the parent module of Autodiff
    # const parent_mod = Base.module_parent(current_module())
    const parent_mod = Main  # temporarily, for prototyping

	export getSymbols, substSymbols, diff, dprefix
	export @deriv_rule, deriv_rule, declareType
	export newvar, resetvar

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

		# vhint					  # stores all expression values to match adequate derivation rule

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

	##### now include parsing and derivation scripts
	include("deriv_rules.jl")
	include("pass1.jl")
	include("pass2.jl")
	include("diff.jl")


