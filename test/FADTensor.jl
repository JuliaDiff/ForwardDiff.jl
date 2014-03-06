using ForwardDiff
using Base.Test

args = FADTensor([1., 2., 3.])

f(x, y, z) = x*x*x+y*y*y+z*z*z+x*y*y+y*z*z+x*z*z
#f(x, y, z) = x^3+y^3+z^3+x*y^2+y*z^2+x*z^2

z = f(args...)
#println("f(x, y, z) = x^3+y^3+z^3+x*y^2+y*z^2+x*z^2\n")

@test value(z) == 67.
#println("f(1, 2, 3) = ", value(z))

@test grad(z) == [16, 25, 45]
#println("Gradient of f:\n", grad(z))

@test hessian(z) == [6 4 6; 4 14 6; 6 6 24]
#println("Hessian H(f):\n", hessian(z))

tz = tensor(z)
@test tz[:, :, 1] == [6 0 0; 0 2 0; 0 0 2]
@test tz[:, :, 2] == [0 2 0; 2 6 0; 0 0 2]
@test tz[:, :, 3] == [0 0 2; 0 0 2; 2 2 6]
#println("Tensor of f:\n", tensor(z))
