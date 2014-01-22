######### evaluates once all variables to give type hints for derivation ############
#  most gradient calculation statements depend on the type of variables (Scalar or Array)
#  this is where they are evaluated (with values stored in global Dict 'vhint' )
function preCalculate(m::ParsingStruct)
    global vhint = Dict()

    body = Expr[ [ :( $(p[1]) = $(p[2]) ) for p in zip(m.insyms, m.init)]..., 
                 m.exprs...]
    
    vl = getSymbols(body)  # list of all vars (external, parameters, set by model, and accumulator)
    body = vcat(body, 
    			[ :(vhint[$(Expr(:quote, v))] = $v) for v in vl ])

	# identify external vars and add definitions x = Main.x
	header = [ :( local $v = $(Expr(:., :Main, Expr(:quote, v))) ) for v in external(m)]

	# build and evaluate the let block containing the function and external vars hooks
	# Note that evaluation takes place in the parent module (where extra functions are defined)
	try
		vhint = eval(parent_mod, Expr(:let, Expr(:block, vcat(header, :(vhint=Dict() ), body, :vhint)...) ))
	catch e
		rethrow(e)
		# error("Model fails to evaluate for initial values given")
	end

	res = vhint[m.outsym]
	!isa(res, Real) && error("Model outcome should be a scalar, $(typeof(res)) found")
	# res == -Inf && error("Initial values out of model support, try other values")
end

######### builds the gradient expression from unfolded expression ##############
function backwardSweep!(m::ParsingStruct)  

	explore(ex::Expr) = explore(toExH(ex))
	explore(ex::ExH) = error("[backwardSweep] unmanaged expr type $(ex.head)")
	explore(ex::ExLine) = nothing

	function explore(ex::ExEqual)
		lhs = ex.args[1]
		isSymbol(lhs) || isRef(lhs) || error("[backwardSweep] not a symbol / ref on LHS of assigment $(ex)")
		dsym = lhs
		dsym2 = dprefix(lhs)
		
		rhs = ex.args[2]
		if !isSymbol(rhs) && !isa(rhs,Expr) # some kind of number, nothing to do

		elseif isSymbol(rhs) 
			if in(rhs, avars)
				vsym2 = dprefix(rhs)
				push!(m.dexprs, :( $vsym2 = $dsym2 ))
			end

		elseif isRef(rhs)
			if in(rhs.args[1], avars)
				vsym2 = dprefix(rhs)
				push!(m.dexprs, :( $vsym2 = $dsym2))
			end

		elseif isDot(rhs)
			if in(rhs.args[1], avars)
				# println("$(rhs.args[1]) -> $(:( getfield( $(rhs.args[1]), $(Expr(:quote, rhs.args[2])) ) ))  / $dsym ")
				m.dexprs = vcat(m.dexprs, derive(:( getfield( $(rhs.args[1]), $(Expr(:quote, rhs.args[2])) ) ), 1, dsym))
			end

		elseif isa(toExH(rhs), ExCall)  
			for i in 2:length(rhs.args) 
				vsym = rhs.args[i]
				if isa(vsym, Symbol) && in(vsym, avars)
					m.dexprs = vcat(m.dexprs, derive(rhs, i-1, dsym))
				end
			end
		else 
			error("[backwardSweep] can't derive $rhs")
		end
	end

	avars = activeVars(m)
	m.dexprs = Expr[]
	for ex2 in reverse(m.exprs)  # proceed backwards
		isa(ex2, Expr) || error("[backwardSweep] not an expression : $ex2")
		explore(ex2)
	end
end


####   Applies derivation rule to statement  ####

## returns sample value for the given Symbol or Expr (for refs)
hint(v::Symbol) = vhint[v]
hint(v) = v  # should be a value if not a Symbol or an Expression
function hint(v::Expr)
	if isRef(v) 
		v.args[1] = :( vhint[$(Expr(:quote, v.args[1]))] )
		return eval(v)
	elseif v.head == :quote  # argument is a symbol (getfield)
		return v.args[1]
	else
		error("[hint] unexpected variable $v ($(v.head))")
	end
end

#########   Returns gradient expression of opex       ###########
function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))  # opex=:(z^x);index=2;dsym=:y
	vs = opex.args[1+index]
	ds = dprefix(dsym)
	args = opex.args[2:end]
	
	val = map(hint, args)  # get sample values of args to find correct gradient statement

	fn = symbol("d_$(opex.args[1])_x$index")

	try
		dexp = eval(Expr(:call, fn, val...))

		smap = { symbol("x$i") => args[i] for i in 1:length(args)}
		smap[:ds] = ds
		smap[:ds1] = symbol("$(ds).1")
		smap[:ds2] = symbol("$(ds).2")
		smap[:drec] = dprefix(vs)
		smap[:drec1] = dprefix("$(vs).1")
		smap[:drec2] = dprefix("$(vs).2")
		dexp = substSymbols(dexp, smap)

		return dexp
	catch e 
		error("[derive] Failed to derive $opex by argument $vs ($(map(typeof, val)))")
	end

end