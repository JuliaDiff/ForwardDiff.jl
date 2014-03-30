using ForwardDiff
using Base.Test

### Testing functions f:R^n->R

# Testing addition, subtraction, scalar multiplication, multiplication and integer powers

f(x, y, z) = x^6-y^5+z^2+x^4*y*z-3*x*y^3*z^2+2*x^2*y^3-4*y^2*z+y*z^2-2*x*y*z

dfdx(x, y, z) = 6*x^5+4*x^3*y*z-3*y^3*z^2+4*x*y^3-2*y*z
dfdy(x, y, z) = -5*y^4+x^4*z-9*x*y^2*z^2+6*x^2*y^2-8*y*z+z^2-2*x*z
dfdz(x, y, z) = 2*z+x^4*y-6*x*y^3*z-4*y^2+2*y*z-2*x*y
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

args = [2.3, -1.5, -4.]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing the API

f(x) =
  x[1]^6-x[2]^5+x[3]^2+x[1]^4*x[2]*x[3]-3*x[1]*x[2]^3*x[3]^2+2*x[1]^2*x[2]^3-4*x[2]^2*x[3]+x[2]*x[3]^2-2*x[1]*x[2]*x[3]

dfdx(x) = 6*x[1]^5+4*x[1]^3*x[2]*x[3]-3*x[2]^3*x[3]^2+4*x[1]*x[2]^3-2*x[2]*x[3]
dfdy(x) = -5*x[2]^4+x[1]^4*x[3]-9*x[1]*x[2]^2*x[3]^2+6*x[1]^2*x[2]^2-8*x[2]*x[3]+x[3]^2-2*x[1]*x[3]
dfdz(x) = 2*x[3]+x[1]^4*x[2]-6*x[1]*x[2]^3*x[3]-4*x[2]^2+2*x[2]*x[3]-2*x[1]*x[2]
gradf(x) = [dfdx(x), dfdy(x), dfdz(x)]

args = [2.3, -1.5, -4.]
output = f(GraDual(args))

@test_approx_eq value(output) f(args)
@test_approx_eq grad(output) gradf(args)

g = forwarddiff_gradient(f, Float64, fadtype=:typed)
output = g(args)
@test_approx_eq output gradf(args)

# Testing division

f(x, y, z) = x^5*z/y^4-1/z

dfdx(x, y, z) = 5*z*x^4/y^4
dfdy(x, y, z) = -4*x^5*z/y^5
dfdz(x, y, z) = x^5/y^4+1/z^2
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

args = [2., 5., -1.]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing square roots, qubic roots and rational powers

f(x, y, z) = z^4*sqrt(x)/cbrt(y)+x^(-5//3)

dfdx(x, y, z) = z^4/(2*sqrt(x)*cbrt(y))-5//3*x^(-8//3)
dfdy(x, y, z) = -z^4*sqrt(x)/(3*y^(4//3))
dfdz(x, y, z) = 4*z^3*sqrt(x)/cbrt(y)
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

args = [0.75, 2.5, -1.25]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing floating and functional powers

a = 4.75
f(x, y, z) = sqrt(x)^(y^4)+cbrt(y)*z^a

dfdx(x, y, z) = x^(y^4/2-1)*y^4/2
dfdy(x, y, z) = 4*sqrt(x)^(y^4)*y^3*log(sqrt(x))+z^a/(3*y^(2/3))
dfdz(x, y, z) = a*cbrt(y)*z^(a-1)
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

args = [1.25, 0.5, 1.5]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing exp, log, log2 and and log10 

f(x, y, z) = log2(x)*exp(y*z)+log(x^4*y)/log10(z)

dfdx(x, y, z) = (exp(y*z)/log(2)+4*log(10)/log(z))/x
dfdy(x, y, z) = exp(y*z)*z*log(x)/log(2)+log(10)/(y*log(z))
dfdz(x, y, z) = exp(y*z)*y*log(x)/log(2)-log(10)*log(x^4*y)/(z*log(z)^2)
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

args = [0.3, 1.1, 2.25]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing trigonometric functions

f(x, y, z) = sin(x*y)+cos(y*z)-tan(x+z)

dfdx(x, y, z) = y*cos(x*y)-sec(x+z)^2
dfdy(x, y, z) = x*cos(x*y)-z*sin(y*z)
dfdz(x, y, z) = -sec(x+z)^2-y*sin(y*z)
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

args = [1.1, -0.23, -2.1]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing inverse trigonometric functions

f(x) = asin(x)
gradf(x) = 1/sqrt(1-x^2)

args = [-0.51]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

f(x) = acos(x)
gradf(x) = -1/sqrt(1-x^2)

args = [0.6]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

f(x) = atan(x)
gradf(x) = 1/(1+x^2)

args = [-0.73]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing hyperbolic functions

f(x) = sinh(x)
gradf(x) = cosh(x)

args = [1.5]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

f(x) = cosh(x)
gradf(x) = sinh(x)

args = [2.35]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

f(x) = tanh(x)
gradf(x) = sech(x)^2

args = [3.52]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

# Testing inverse hyperbolic functions

f(x) = asinh(x)
gradf(x) = 1/sqrt(1+x^2)

args = [1.25]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

f(x) = acosh(x)
gradf(x) = 1/(sqrt(x-1)*sqrt(x+1))

args = [1.12]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)

f(x) = atanh(x)
gradf(x) = 1/(1-x^2)

args = [-0.57]
output = f(GraDual(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
