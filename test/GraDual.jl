## Calculation of f(2, 5), df(2, 5)/dx and df(2, 5)/dy for function 
## f:R^2->R^2 given by f(x, y) = [3*x*y, y^2]

using ForwardDiff
using Base.Test

args = gradual([2., 5.])

f(x, y) = [3*x*y, y^2]

y = f(args...)
#println("f(x, y) = [3*x*y, y^2]\n")

@test value(y) == [30, 25]
#println("f(2, 5) = \n", value(y))

@test jacobian(y) == [15 6; 0 10]
#println("Jacobian J(f):\n", jacobian(y))
