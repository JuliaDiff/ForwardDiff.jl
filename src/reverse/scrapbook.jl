include("p:/documents/julia/AD.jl/src/Autodiff.jl")


tex = :( res = x + y )
Autodiff.diff(tex, :res, x=1.0, y=2.0)


ex = :( res = x[1]^2 + (x[2]-2x[3])^4 )
Autodiff.diff(ex, :res, x=zeros(3))


foo(x::Float64) = sin(x) * exp(-x*x)
Autodiff.@deriv_rule  foo(x) x  dx += ( cos(x)*exp(-x*x) - 2x * sin(x) * exp(-x*x) ) * ds

tex2 = :( res = foo(x) )
ex1, ex2, outsym = Autodiff.diff(tex2, :res, x=1.0)

fex = :( foofoo(x::Float64) = ( $ex2 ; (res, dx) ) )
eval(fex)

foofoo(1.0)

delta = 1e-4
x0 = -.2
[ foofoo(x0) ( foofoo(x0+delta)[1]-foofoo(x0)[1] ) / delta ]


# slight transformation  :dx -> :ex, :dres -> :eres
ex2bis = quote 
    res = foo(x)
    eres = 1.0
    ex = 0.0 * res
    ex += *(-(*(cos(x),exp(*(-(x),x))),*(*(2,x),sin(x),exp(*(-(x),x)))),eres)
    res2 = ex
end

ex12, ex22, outsym2 = Autodiff.diff(ex2bis, :res2, x=1.0)


Set{Any}(:tmp_5,:tmp_14,:tmp_12,:tmp_6,:tmp_10,:tmp_9,:tmp_3,:tmp_11,
	:x,:tmp_4,:tmp_13,:tmp_1,:tmp_7,:ex_1,:tmp_2,:tmp_8,:res2)


fex = :( foofoo2(x::Float64) = ( $ex22 ; (res, res2, dx) ) )
eval(fex)

[foofoo(1.0), foofoo2(1.0)]

delta = 1e-4
x0 = -0.2
[ foofoo2(x0) ( foofoo2(x0+delta)[2]-foofoo2(x0)[2] ) / delta ]



############## zoom
ag = 
[:tmp_18=>Set{Symbol}(:tmp_17),
:tmp_26=>Set{Symbol}(:tmp_20,:tmp_25),
:tmp_20=>Set{Symbol}(:x),
:ex=>Set{Symbol}(),
:tmp_22=>Set{Symbol}(:x),
:eres=>Set{Symbol}(),
:ex_2=>Set{Symbol}(:tmp_28,:ex),
:tmp_25=>Set{Symbol}(:tmp_21,:tmp_24),
:tmp_27=>Set{Symbol}(:tmp_19,:tmp_26),
:tmp_15=>Set{Symbol}(:x),
:tmp_16=>Set{Symbol}(:x),
:tmp_21=>Set{Symbol}(:x),
:res=>Set{Symbol}(:x),
:tmp_28=>Set{Symbol}(:eres,:tmp_27),
:tmp_19=>Set{Symbol}(:tmp_18,:tmp_15),
:tmp_23=>Set{Symbol}(:tmp_22,:x),
:tmp_24=>Set{Symbol}(:tmp_23),
:tmp_17=>Set{Symbol}(:tmp_16,:x),
:res2=>Set{Symbol}(:ex_2)]
ag
intersect(union(Set(m.outsym), Autodiff.relations(:res2, ag)), 
		                                     union(Set(m.insyms...), relations(m.insyms, m.dg))


#dg
[:tmp_26=>Set{Symbol}(:tmp_27),
:tmp_18=>Set{Symbol}(:tmp_19),
:tmp_20=>Set{Symbol}(:tmp_26),
:ex=>Set{Symbol}(:ex_2),
:tmp_22=>Set{Symbol}(:tmp_23),
:eres=>Set{Symbol}(:tmp_28),
:ex_2=>Set{Symbol}(:res2),
:tmp_25=>Set{Symbol}(:tmp_26),
:tmp_27=>Set{Symbol}(:tmp_28),
:x=>Set{Symbol}(:tmp_16,:tmp_23,:tmp_22,:tmp_21,:tmp_17,:res,:tmp_20,:tmp_15),
:tmp_15=>Set{Symbol}(:tmp_19),
:tmp_16=>Set{Symbol}(:tmp_17),
:tmp_21=>Set{Symbol}(:tmp_25),
:tmp_28=>Set{Symbol}(:ex_2),
:tmp_19=>Set{Symbol}(:tmp_27),
:tmp_23=>Set{Symbol}(:tmp_24),
:tmp_24=>Set{Symbol}(:tmp_25),
:tmp_17=>Set{Symbol}(:tmp_18)]

[:ex=>:ex_2]:(res = foo(x))




foofoo2(x::Float64) = begin 
begin 
	res = foo(x)
	eres = 1.0
	ex = 0.0
	tmp_1 = cos(x)
	tmp_2 = -(x)
	tmp_3 = *(tmp_2,x)
	tmp_4 = exp(tmp_3)
	tmp_5 = *(tmp_1,tmp_4)
	tmp_6 = *(2,x)
	tmp_7 = sin(x)
	tmp_8 = -(x)
	tmp_9 = *(tmp_8,x)
	tmp_10 = exp(tmp_9)
	tmp_11 = *(tmp_7,tmp_10)
	tmp_12 = *(tmp_6,tmp_11)
	tmp_13 = -(tmp_5,tmp_12)
	tmp_14 = *(tmp_13,eres)
	ex_1 = +(ex,tmp_14)
	res2 = ex_1
	dtmp_5 = 0.0
	dtmp_14 = 0.0
	dtmp_12 = 0.0
	dtmp_6 = 0.0
	dtmp_10 = 0.0
	dtmp_9 = 0.0
	dtmp_3 = 0.0
	dtmp_11 = 0.0
	dx = 0.0
	dtmp_4 = 0.0
	dtmp_13 = 0.0
	dtmp_1 = 0.0
	dtmp_7 = 0.0
	dex_1 = 0.0
	dtmp_2 = 0.0
	dtmp_8 = 0.0
	dres2 = 1.0
	dex_1 = dres2
	dtmp_14 += dex_1
	dtmp_13 += *(eres,dtmp_14)
	dtmp_5 += dtmp_13
	dtmp_12 -= dtmp_13
	dtmp_6 += *(tmp_11,dtmp_12)
	dtmp_11 += *(tmp_6,dtmp_12)
	dtmp_7 += *(tmp_10,dtmp_11)
	dtmp_10 += *(tmp_7,dtmp_11)
	dtmp_9 += *(exp(tmp_9),dtmp_10)
	dtmp_8 += *(x,dtmp_9)
	dx += *(tmp_8,dtmp_9)
	dx -= dtmp_8
	dx += *(cos(x),dtmp_7)
	dx += *(2,dtmp_6)
	dtmp_1 += *(tmp_4,dtmp_5)
	dtmp_4 += *(tmp_1,dtmp_5)
	dtmp_3 += *(exp(tmp_3),dtmp_4)
	dtmp_2 += *(x,dtmp_3)
	dx += *(tmp_2,dtmp_3)
	dx -= dtmp_2
	dx -= *(sin(x),dtmp_1)
	dx += *(-(*(cos(x),exp(*(-(x),x))),*(*(2,x),sin(x),exp(*(-(x),x)))),dres)
end
(res,res2,dx)
end



################################################

function bar(x::Vector{Float64})
    tmp_36 = x[1]
    tmp_37 = ^(tmp_36,2)
    tmp_38 = x[2]
    tmp_39 = x[3]
    tmp_40 = *(2,tmp_39)
    tmp_41 = -(tmp_38,tmp_40)
    tmp_42 = ^(tmp_41,4)
    res = +(tmp_37,tmp_42)
    dtmp_37 = 0.0
    dtmp_41 = 0.0
    dtmp_38 = 0.0
    dres = 1.0
    dtmp_36 = 0.0
    dtmp_39 = 0.0
    fill!(dx,0.0)
    dtmp_42 = 0.0
    dtmp_40 = 0.0
    dtmp_37 += dres
    dtmp_42 += dres
    dtmp_41 += *(4,^(tmp_41,-(4,1)),dtmp_42)
    dtmp_38 += dtmp_41
    dtmp_40 -= dtmp_41
    dtmp_39 += *(2,dtmp_40)
    dx[3] = dtmp_39
    dx[2] = dtmp_38
    dtmp_36 += *(2,^(tmp_36,-(2,1)),dtmp_37)
    dx[1] = dtmp_36

    (res, dx)
end

dx = Array(Float64,(3,))
bar([0.,0,1.])
bar([2.,0,0])


