##################################################################
#  Main function definition
############################################################################

function diff(model::Expr, out::Symbol, skipgradient=false; init...)

   	length(init)>=1 || error("There should be at least one parameter specified, none found")

	m = ParsingStruct()
	m.source = model
	m.outsym = out
	m.insyms = map(sv->sv[1], init)
	m.init = map(sv->sv[2], init)

	unfold!(m)	

	m.ag, m.dg, subst, m.exprs = varGraph(m.exprs)
	println(m.ag, m.dg, subst, m.exprs)

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
		println(avars)

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
				push!(header, :( local $dsym = Array(Float64, $(Expr(:tuple,size(vh)...)))) )
				push!(body, :( fill!($dsym, 0.) ) )
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

