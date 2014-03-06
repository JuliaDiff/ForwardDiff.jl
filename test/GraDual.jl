## Calculation of f(2, 5), df(2, 5)/dx and df(2, 5)/dy for function 
## f:R^2->R^2 given by f(x, y) = [3*x*y, y^2]

using ForwardDiff
using Base.Test

args = GraDual([2., 5.])

f(x, y) = [3*x*y, y^2]
#f(x) = [3*x[1]*x[2], x[2]^2] # works too

args = GraDual([2., 5.])

y = f(args...)
#println("f(x, y) = [3*x*y, y^2]\n")

@test value(y) == [30, 25]
#println("f(2, 5) = \n", value(y))

@test jacobian(y) == [15 6; 0 10]
#println("Jacobian J(f):\n", jacobian(y))

# Testing the API
g = forwarddiff_jacobian(f, Float64, n=2, fadtype=:typed)
out = g([2., 5.])
@test out == [15 6; 0 10]
