## Calculation of f(2, 5) and df(2, 5)/dx and df(2, 5)/dy for function 
## f:R^2->R^2 given by f(x, y) = [3*x*y, y^2]

using AutoDiff

args = gradual([2., 5.])

f(x, y) = [3*x*y, y^2]

y = f(args...)

println("f(x, y) = [3*x*y, y^2]\n")
println("f(2, 5) = ", value(y), "\n")
#grad(y))
println("Jacobian J(f):\n", jacobian(y))
