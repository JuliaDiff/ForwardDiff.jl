using ForwardDiff
using Base.Test

### Testing functions f:R^n->R

# Testing addition, subtraction, scalar multiplication, multiplication and integer powers

f(x, y, z) = x^3-y^3+z^3+x^2*y-5*x*z^2+x*y*z
dfdx(x, y, z) = 3*x^2+2*x*y-5*z^2+y*z
dfdy(x, y, z) = -3*y^2+x^2+x*z
dfdz(x, y, z) = 3*z^2-10*x*z+x*y

args = [2., 5., -1.]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) [dfdx(args...), dfdy(args...), dfdz(args...)]

# Testing the API
g = forwarddiff_jacobian(f, Float64, n=3, fadtype=:typed)
output = g(args)
@test_approx_eq output [dfdx(args...), dfdy(args...), dfdz(args...)]
