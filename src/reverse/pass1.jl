
######## unfolds expressions to prepare derivation ###################
function unfold!(m::ParsingStruct)

	explore(ex::Expr)           = explore(toExH(ex))
	explore(ex::ExH)            = error("[unfold] unmanaged expr type $(ex.head) in ($ex)")
	explore(ex::ExLine)         = nothing     # remove line info
	explore(ex::LineNumberNode) = nothing     # remove line info
	explore(ex::ExRef)          = toExpr(ex)  # unchanged
	explore(ex::ExComp)         = toExpr(ex)  # unchanged
	explore(ex::ExVcat)         = explore(Expr(:call, :vcat, ex.args...) )  # translate to vcat() call, and explore
	explore(ex::ExTrans)        = explore(Expr(:call, :transpose, ex.args[1]) )  # translate to transpose() and explore
	explore(ex::ExDot)          = toExpr(ex)   # unchanged
	explore(ex::ExPEqual)       = (args = ex.args ; explore( Expr(:(=), args[1], Expr(:call, :+, args[1], args[2])) ) )
	explore(ex::ExMEqual)       = (args = ex.args ; explore( Expr(:(=), args[1], Expr(:call, :-, args[1], args[2])) ) )
	explore(ex::ExTEqual)       = (args = ex.args ; explore( Expr(:(=), args[1], Expr(:call, :*, args[1], args[2])) ) )
	explore(ex::Any)            = ex

	explore(ex::ExBlock) = map( ei -> (re = explore(ei) ; re==nothing || push!(m.exprs, re)), ex.args )

	function explore(ex::ExEqual) 
		lhs = ex.args[1]
		isSymbol(lhs) || isRef(lhs) || error("[unfold] not a symbol on LHS of assigment $ex")

		rhs = ex.args[2]
		if isSymbol(rhs) || isa(rhs, Real) || isDot(rhs)
			push!(m.exprs, Expr(:(=), lhs, rhs))
		elseif isa(rhs, Expr) 
			ue = explore(toExH(rhs)) # explore will return something in this case
			push!(m.exprs, Expr(:(=), lhs, ue))
		else  # unmanaged kind of rhs
			error("[unfold] can't handle RHS of assignment $(toExpr(ex))")
		end
		return nothing
	end

	function explore(ex::ExCall) 
		na = {ex.args[1]}   # function name
		args = ex.args[2:end]  # arguments

		# if more than 2-ary call, convert to nested 2-ary calls
		#  (easier for derivation), applies to +, sum, *, min, max
		# TODO : check if applicable to other n-ary (n>2) operators
		if in(na[1], [:+, :*, :sum, :min, :max]) 
			while length(args) > 2
				a2 = pop!(args)
				a1 = pop!(args)
				push!(args, Expr(:call, ex.args[1], a1, a2))
			end
		end

		for e2 in args  
			if isa(e2, Expr) # only refs and calls will work
				ue = explore(e2)
				nv = newvar(TEMP_NAME)
				push!(m.exprs, :($nv = $ue))
				push!(na, nv)
			else
				push!(na, e2)
			end
		end

		Expr(ex.head, na...)
	end

	m.exprs = Expr[]
	explore(m.source)
end

######### analyzes and transforms for derivation #############
# - makes variables set several times unique (necessary for back propagation)
# - processes functions that transform their arguments (copy!, gemm!, etc...)  # TODO : unfinished !
# - builds the variable dependency graph
# FIXME : algo doesn't work when assigning on individual elements of an array, x = .. then x[3] = ...; 
function varGraph(vex::Vector{Expr})
	subst = Dict{Symbol, Symbol}()     # will store variable renamings
	touched = Set{Symbol}()            # variables set
	external = Set{Symbol}()           # variables defined outside
	vg = Dict{Symbol, Set{Symbol}}()   # dependency graph
	nvex = Expr[]                      # returned transformed vector of expressions


	explore(ex::Expr) =    explore(toExH(ex))
	explore(ex::ExH) =     error("[varGraph!] unmanaged expr type $(ex.head) in ($ex)")
	
	function explore(ex::Union(ExCall, ExEqual))

		ex = substSymbols(ex, subst) # first, do renaming
		if isa(ex, ExCall)
			#  where to look for changed variable, TODO : generalize, make user settable
			const inplace_var = {:copy! => 1, :gemm! => 7 }

			fn = ex.args[1]
			haskey(inplace_var, fn) || error("[varGraph!] unknown function $(ex.args[1])")

			fa = ex.args[2:end]
			lhss = fa[ get(inplace_var, fn, 1) ]
			isSymbol(lhss) || error("[varGraph!] expected symbol got $lhss in $ex")

			rhss = getSymbols( fa[ 1:length(fa) .!= get(inplace_var, fn, 1) ] )  # FIXME : not true for gemm!
		else # assigment case
			lhs = ex.args[1]
			lhss = isSymbol(lhs) ? lhs : lhs.args[1]  # extract only symbol if ref
			rhss = getSymbols( ex.args[2] )
		end
		
		external = union(external, {setdiff(rhss, touched)...})
		in(lhss, external) && error("$lhss is both an external variable and a variable set by the model")

		if in(lhss, touched) # ex is setting an already set variable => new var creation
			ss = in(lhss, values(subst)) ? (collect(keys(subst)))[(findin(collect(values(subst)), [lhss]))[1]] : lhss
			nv = newvar(ss)
			subst[ss] = nv   # generate new name, add it to substitution list for following statements
			subst[lhss] = nv #  for previous symbol too

			if isa(ex, ExCall) # different var renaming between assigment and calls
		        ex.args[2:end] = substSymbols(fa, subst) # replace in lhs
		        push!(nvex, :( $(subst[lhss]) = similar($lhss) ) ) # need to allocate variable in this case
		    else
	        	ex.args[1] = substSymbols(lhs, subst) # replace in lhs
			end
		else
			nv = lhss
	    end

		push!(touched, nv)  # add to touched variable set
		push!(nvex, ex)     # add to transformed expression vector
		vg[nv] = rhss       # update dependency graph
	end

	map(explore, vex)

	# invert dependency graph
	vgi = Dict{Symbol, Set}()
	for (k,v) in vg
		for s in v
			haskey(vgi,s) ? push!(vgi[s], k) : (vgi[s] = Set(k))
		end
	end

	(vg, vgi, subst, nvex)
end
