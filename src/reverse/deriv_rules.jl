##########################################################################################
#
#    Derivation rules & function 'derive' returning the expr of gradient
#
##########################################################################################
# TODO : add operators : hcat, vcat, ? : , map, mapreduce, if else 

#########   function to declare a new type in Autodiff (for extensibility)  ######
declareType(a::Type, na::Symbol) = eval(:( $na = $a ))

#########   macro and function to simplify derivation rules creation  ###########
function deriv_rule(func::Expr, dv::Symbol, diff::Expr)
	argsn = map(e-> isa(e, Symbol) ? e : e.args[1], func.args[2:end])
	index = find(dv .== argsn)[1]

	# change var names in signature and diff expr to x1, x2, x3, ..
	smap = { argsn[i] => symbol("x$i") for i in 1:length(argsn) }
	# symbols for distributions
	smap[ symbol("d$dv")] = symbol("drec") 
	# notation for composite type derivatives
	smap[ symbol("d$(dv)1")] = symbol("drec1")  
	smap[ symbol("d$(dv)2")] = symbol("drec2")

	args2 = substSymbols(func.args[2:end], smap)

	# diff function name
	fn = symbol("d_$(func.args[1])_x$index")

	fullf = Expr(:(=), Expr(:call, fn, args2...), Expr(:quote, substSymbols(diff, smap)) )
	eval(fullf)
end

# macro version
macro deriv_rule(func::Expr, dv::Symbol, diff::Expr)
	deriv_rule(func, dv, diff)
end


########   rules definitions   #############

# addition
@deriv_rule +(x::Real         , y::Real )            x     dx += ds
@deriv_rule +(x::Real         , y::AbstractArray)    x     for i in 1:length(ds) ; dx += ds[i]     ;end
@deriv_rule +(x::AbstractArray, y       )            x     for i in 1:length(ds) ; dx[i] += ds[i]  ;end
@deriv_rule +(x::Real         , y::Real )            y     dy += ds
@deriv_rule +(x::AbstractArray, y::Real )            y     for i in 1:length(ds) ; dy    += ds[i]  ;end
@deriv_rule +(x               , y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] += ds[i]  ;end

# unary substraction
@deriv_rule -(x::Real )              x     dx -= ds
@deriv_rule -(x::AbstractArray)      x     for i in 1:length(ds) ; dx[i] -= ds[i]  ;end

# binary substraction
@deriv_rule -(x::Real         , y::Real )            x     dx += ds
@deriv_rule -(x::Real         , y::AbstractArray)    x     for i in 1:length(ds) ; dx += ds[i]     ;end
@deriv_rule -(x::AbstractArray, y       )            x     for i in 1:length(ds) ; dx[i] += ds[i]  ;end
@deriv_rule -(x::Real         , y::Real )            y     dy -= ds
@deriv_rule -(x::AbstractArray, y::Real )            y     for i in 1:length(ds) ; dy    -= ds[i]  ;end
@deriv_rule -(x               , y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] -= ds[i]  ;end

# sum()
@deriv_rule sum(x::Real )           x     dx += ds
@deriv_rule sum(x::AbstractArray)   x     for i in 1:length(x) ; dx[i] += ds     ;end

# dot()
@deriv_rule dot(x::AbstractArray, y::AbstractArray)    x     for i in 1:length(x) ; dx[i] += y[i]*ds ;end
@deriv_rule dot(x::AbstractArray, y::AbstractArray)    y     for i in 1:length(y) ; dy[i] += x[i]*ds ;end

# log() and exp()
@deriv_rule log(x::Real )            x     dx += ds / x
@deriv_rule log(x::AbstractArray)    x     for i in 1:length(ds) ; dx[i] += ds[i] / x[i]  ;end

@deriv_rule exp(x::Real )            x     dx += exp(x) * ds    # TODO : allow :s placeholder for optimization
@deriv_rule exp(x::AbstractArray)    x     for i in 1:length(ds) ; dx[i] += exp(x[i]) * ds[i] ;end

# sin() and cos()
@deriv_rule sin(x::Real )            x     dx += cos(x) * ds
@deriv_rule sin(x::AbstractArray)    x     for i in 1:length(ds) ; dx[i] += cos(x[i]) * ds[i] ;end

@deriv_rule cos(x::Real )            x     dx -= sin(x) * ds
@deriv_rule cos(x::AbstractArray)    x     for i in 1:length(ds) ; dx[i] -= sin(x[i]) * ds[i] ;end

# abs, max(), min()
@deriv_rule abs(x::Real )            x     dx += sign(x) * ds
@deriv_rule abs(x::AbstractArray)    x     for i in 1:length(ds) ; dx[i] += sign(x[i]) * ds[i] ;end

@deriv_rule max(x::Real         , y::Real )   		x     dx += (x > y) * ds
@deriv_rule max(x::Real         , y::AbstractArray)  x     for i in 1:length(ds) ; dx += (x > y[i]) * ds[i] ; end
@deriv_rule max(x::AbstractArray, y::Real )   		x     for i in 1:length(ds) ; dx[i] += (x[i] > y) * ds[i] ; end
@deriv_rule max(x::AbstractArray, y::AbstractArray)	x     for i in 1:length(ds) ; dx[i] += (x[i] > y[i]) * ds[i] ; end
@deriv_rule max(x::Real         , y::Real )   		y     dy += (x < y) * ds
@deriv_rule max(x::Real         , y::AbstractArray)  y     for i in 1:length(ds) ; dy[i] += (x < y[i]) * ds[i] ; end
@deriv_rule max(x::AbstractArray, y::Real )   		y     for i in 1:length(ds) ; dy += (x[i] < y) * ds[i] ; end
@deriv_rule max(x::AbstractArray, y::AbstractArray)  y     for i in 1:length(ds) ; dy[i] += (x[i] < y[i]) * ds[i] ; end

@deriv_rule min(x::Real         , y::Real )          x     dx += (x < y) * ds
@deriv_rule min(x::Real         , y::AbstractArray)  x     for i in 1:length(ds) ; dx += (x < y[i]) * ds[i] ; end
@deriv_rule min(x::AbstractArray, y::Real )          x     for i in 1:length(ds) ; dx[i] += (x[i] < y) * ds[i] ; end
@deriv_rule min(x::AbstractArray, y::AbstractArray)  x     for i in 1:length(ds) ; dx[i] += (x[i] < y[i]) * ds[i] ; end
@deriv_rule min(x::Real         , y::Real )          y     dy += (x > y) * ds
@deriv_rule min(x::Real         , y::AbstractArray)  y     for i in 1:length(ds) ; dy[i] += (x > y[i]) * ds[i] ; end
@deriv_rule min(x::AbstractArray, y::Real )          y     for i in 1:length(ds) ; dy += (x[i] > y) * ds[i] ; end
@deriv_rule min(x::AbstractArray, y::AbstractArray)  y     for i in 1:length(ds) ; dy[i] += (x[i] > y[i]) * ds[i] ; end

# multiplication
@deriv_rule *(x::Real         , y::Real )           x     dx += y * ds
@deriv_rule *(x::Real         , y::AbstractArray)   x     for i in 1:length(ds) ; dx += y[i] * ds[i] ; end
@deriv_rule *(x::AbstractArray, y::Real )           x     for i in 1:length(ds) ; dx[i] += y * ds[i] ; end
@deriv_rule *(x::AbstractArray, y::Vector)          x     gemm!('N', 'T', 1., ds, reshape(y, length(y), 1), 1., dx)  # reshape needed 
@deriv_rule *(x::AbstractArray, y::AbstractArray)   x     gemm!('N', 'T', 1., ds, y, 1., dx)

@deriv_rule *(x::Real         , y::Real )           y     dy += x * ds
@deriv_rule *(x::Real         , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += x * ds[i] ; end
@deriv_rule *(x::AbstractArray, y::Real )           y     for i in 1:length(ds) ; dy += x[i] * ds[i] ; end
@deriv_rule *(x::AbstractArray, y::Vector)          y     gemm!('T', 'N', 1., x, reshape(ds, length(ds), 1), 1., dy)
@deriv_rule *(x::AbstractArray, y::AbstractArray)   y     gemm!('T', 'N', 1., x, ds, 1., dy)

# dot multiplication
@deriv_rule .*(x::Real         , y::Real )           x     dx += y .* ds
@deriv_rule .*(x::Real         , y::AbstractArray)   x     for i in 1:length(ds) ; dx += y[i] * ds[i] ; end
@deriv_rule .*(x::AbstractArray, y::Real )           x     for i in 1:length(ds) ; dx[i] += y * ds[i] ; end
@deriv_rule .*(x::AbstractArray, y::AbstractArray)   x     for i in 1:length(ds) ; dx[i] += y[i] * ds[i] ; end

@deriv_rule .*(x::Real         , y::Real )           y     dy += x * ds
@deriv_rule .*(x::Real         , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += x * ds[i] ; end
@deriv_rule .*(x::AbstractArray, y::Real )           y     for i in 1:length(ds) ; dy += x[i] * ds[i] ; end
@deriv_rule .*(x::AbstractArray, y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += x[i] * ds[i] ; end

# power  (both args reals)
@deriv_rule ^(x::Real, y::Real)  x     dx += y * x ^ (y-1) * ds
@deriv_rule ^(x::Real, y::Real)  y     dy += log(x) * x ^ y * ds

# dot power
@deriv_rule .^(x::Real         , y::Real )            x     dx += y * x ^ (y-1) * ds
@deriv_rule .^(x::Real         , y::AbstractArray)    x     for i in 1:length(ds) ; dx += y[i] * x ^ (y[i]-1) * ds[i] ; end
@deriv_rule .^(x::AbstractArray, y::Real )            x     for i in 1:length(ds) ; dx[i] += y * x[i] ^ (y-1) * ds[i] ; end
@deriv_rule .^(x::AbstractArray, y::AbstractArray)    x     for i in 1:length(ds) ; dx[i] += y[i] * x[i] ^ (y[i]-1) * ds[i] ; end

@deriv_rule .^(x::Real         , y::Real )            y     dy += log(x) * x ^ y * ds
@deriv_rule .^(x::AbstractArray, y::Real )            y     for i in 1:length(ds) ; dy += log(x[i]) * x[i] ^ y * ds[i] ; end
@deriv_rule .^(x::Real         , y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] += log(x) * x ^ y[i] * ds[i] ; end
@deriv_rule .^(x::AbstractArray, y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] += log(x[i]) * x[i] ^ y[i] * ds[i] ; end

# division
@deriv_rule /(x::Real         , y::Real )           x     dx += ds / y
@deriv_rule /(x::Real         , y::AbstractArray)   x     for i in 1:length(ds) ; dx += ds[i] / y[i] ; end
@deriv_rule /(x::AbstractArray, y::Real )           x     for i in 1:length(ds) ; dx[i] += ds[i] / y ; end

@deriv_rule /(x::Real         , y::Real )           y     dy -= x * ds / (y * y)
@deriv_rule /(x::Real         , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] -= x * ds[i] / (y[i]*y[i]) ; end
@deriv_rule /(x::AbstractArray, y::Real )           y     for i in 1:length(ds) ; dy -= x[i] * ds[i] / (y * y); end

# dot division
@deriv_rule ./(x::Real         , y::Real )           x     dx += ds / y
@deriv_rule ./(x::Real         , y::AbstractArray)   x     for i in 1:length(ds) ; dx += ds[i] / y[i] ; end
@deriv_rule ./(x::AbstractArray, y::Real )           x     for i in 1:length(ds) ; dx[i] += ds[i] / y ; end
@deriv_rule ./(x::AbstractArray, y::AbstractArray)   x     for i in 1:length(ds) ; dx[i] += ds[i] / y[i] ; end

@deriv_rule ./(x::Real         , y::Real )           y     dy -= x * ds / (y * y)
@deriv_rule ./(x::Real         , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] -= x * ds[i] / (y[i]*y[i]) ; end
@deriv_rule ./(x::AbstractArray, y::Real )           y     for i in 1:length(ds) ; dy -= x[i] * ds[i] / (y * y); end
@deriv_rule ./(x::AbstractArray, y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] -= x[i] * ds[i] / (y[i] * y[i]); end

# transpose
@deriv_rule transpose(x::Real )           x   dx += ds
@deriv_rule transpose(x::AbstractArray)   x   dx += transpose(ds)
