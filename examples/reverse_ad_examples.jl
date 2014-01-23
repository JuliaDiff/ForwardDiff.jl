
include("../src/AutoDiff.jl")


ex = :( res = x + y )
ex1, ex2, outsym = AutoDiff.reversediff(tex, :res, x=1.0, y=2.0)
ex2
# quote 
#     res = +(x,y)
#     dy = 0.0
#     dres = 1.0
#     dx = 0.0
#     dx += dres
#     dy += dres
# end

ex = :( res = x[1]^2 + (x[2]-2x[3])^4 )
ex1, ex2, outsym = AutoDiff.reversediff(ex, :res, x=zeros(3))   # x is vector
ex2
# quote 
#     tmp_1 = x[1]
#     tmp_2 = ^(tmp_1,2)
#     tmp_3 = x[2]
#     tmp_4 = x[3]
#     tmp_5 = *(2,tmp_4)
#     tmp_6 = -(tmp_3,tmp_5)
#     tmp_7 = ^(tmp_6,4)
#     res = +(tmp_2,tmp_7)
#     dtmp_5 = 0.0
#     dtmp_4 = 0.0
#     dtmp_3 = 0.0
#     dtmp_2 = 0.0
#     dres = 1.0
#     dtmp_1 = 0.0
#     dtmp_7 = 0.0
#     fill!(dx,0.0)
#     dtmp_6 = 0.0
#     dtmp_2 += dres
#     dtmp_7 += dres
#     dtmp_6 += *(4,^(tmp_6,-(4,1)),dtmp_7)
#     dtmp_3 += dtmp_6
#     dtmp_5 -= dtmp_6
#     dtmp_4 += *(2,dtmp_5)
#     dx[3] = dtmp_4
#     dx[2] = dtmp_3
#     dtmp_1 += *(2,^(tmp_1,-(2,1)),dtmp_2)
#     dx[1] = dtmp_1
# end


#  AutoDiff comes with only basic functions derivation defined
#  The user can supply additionnal functions
foo(x) = sin(x) * exp(-x*x)

# tell reversediff() how to handle foo when deriving against x
AutoDiff.@deriv_rule  foo(x) x  dx += ( cos(x)*exp(-x*x) - 2x * sin(x) * exp(-x*x) ) * ds

ex = quote
	y = foo(x)
	z = sin(y)
	res = log(z)
end

ex1, ex2, outsym = AutoDiff.reversediff(ex, :res, x=1.0)
ex2
# quote 
#     y = foo(x)
#     z = sin(y)
#     res = log(z)
#     dz = 0.0
#     dy = 0.0
#     dres = 1.0
#     dx = 0.0
#     dz += /(dres,z)
#     dy += *(cos(y),dz)
#     dx += *(-(*(cos(x),exp(*(-(x),x))),*(*(2,x),sin(x),exp(*(-(x),x)))),dy)
# end

# now build a working function
@eval function bar(x)
		$ex2
		(res, dx)
	end

	
bar(1.0)
#(-1.1886262943182573,-1.314252857209243)

# check that d(bar)/dx is correct
[ bar(1.0)[2]  (bar(1.001)[1]-bar(1.)[1]) / 0.001 ]
