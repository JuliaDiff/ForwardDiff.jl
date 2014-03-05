using ForwardDiff
using Base.Test

args = FADHessian([1., 2., 3.])

f(x, y, z) = x^3+y^3+z^3+x*y^2+y*z^2+x*z^2

z = f(args...)
#println("f(x, y, z) = x^3+y^3+z^3+x*y^2+y*z^2+x*z^2\n")

@test value(z) == 67.
#println("f(1, 2, 3) = ", value(z))

@test grad(z) == [16, 25, 45]
#println("Gradient of f:\n", grad(z))

@test hessian(z) == [6 4 6; 4 14 6; 6 6 24]
#println("Hessian H(f):\n", hessian(z))
