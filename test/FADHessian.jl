using ForwardDiff
using Base.Test

### Testing functions f:R^n->R

# Testing addition, subtraction, scalar multiplication, multiplication and integer powers

f(x, y, z) = x^6-y^5+z^2+x^4*y*z-3*x*y^3*z^2+2*x^2*y^3-4*y^2*z+y*z^2-2*x*y*z

dfdx(x, y, z) = 6*x^5+4*x^3*y*z-3*y^3*z^2+4*x*y^3-2*y*z
dfdy(x, y, z) = -5*y^4+x^4*z-9*x*y^2*z^2+6*x^2*y^2-8*y*z+z^2-2*x*z
dfdz(x, y, z) = 2*z+x^4*y-6*x*y^3*z-4*y^2+2*y*z-2*x*y
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

dfdxx(x, y, z) = 30*x^4+12*x^2*y*z+4*y^3
dfdxy(x, y, z) = 4*x^3*z-9*y^2*z^2+12*x*y^2-2*z
dfdyy(x, y, z) = -20*y^3-18*x*y*z^2+12*x^2*y-8*z
dfdxz(x, y, z) = 4*x^3*y-6*y^3*z-2*y
dfdyz(x, y, z) = x^4-18*x*y^2*z-8*y+2*z-2*x
dfdzz(x, y, z) = 2-6*x*y^3+2*y
function hessianf{T<:Real}(x::T, y::T, z::T)
  w = zeros(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

args = [2.3, -1.5, -4.]
output = f(FADHessian(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)

# Testing the API

g = forwarddiff_hessian(f, Float64, fadtype=:typed)
output = g(args)
@test_approx_eq output hessianf(args...)

# Testing division

f(x, y, z) = x^5*z/y^4-1/z

dfdx(x, y, z) = 5*z*x^4/y^4
dfdy(x, y, z) = -4*x^5*z/y^5
dfdz(x, y, z) = x^5/y^4+1/z^2
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

dfdxx(x, y, z) = 20*z*x^3/y^4
dfdxy(x, y, z) = -20*z*x^4/y^5
dfdyy(x, y, z) = 20*x^5*z/y^6
dfdxz(x, y, z) = 5*x^4/y^4
dfdyz(x, y, z) = -4*x^5/y^5
dfdzz(x, y, z) = -2/z^3
function hessianf{T<:Real}(x::T, y::T, z::T)
  w = zeros(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

args = [2., 5., -1.]
output = f(FADHessian(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)

# Testing square roots, qubic roots and rational powers

f(x, y, z) = z^4*sqrt(x)/cbrt(y)+x^(-5//3)

dfdx(x, y, z) = z^4/(2*sqrt(x)*cbrt(y))-5//3*x^(-8//3)
dfdy(x, y, z) = -z^4*sqrt(x)/(3*y^(4//3))
dfdz(x, y, z) = 4*z^3*sqrt(x)/cbrt(y)
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

dfdxx(x, y, z) = -z^4/(4*x^(3/2)*cbrt(y))+40/(9*x^(11/3))
dfdxy(x, y, z) = -z^4/(6*sqrt(x)*y^(4/3))
dfdyy(x, y, z) = 4*z^4*sqrt(x)/(9*y^(7/3))
dfdxz(x, y, z) = 2*z^3/(sqrt(x)*cbrt(y))
dfdyz(x, y, z) = -4*z^3*sqrt(x)/(3*y^(4/3))
dfdzz(x, y, z) = 12*z^2*sqrt(x)/cbrt(y)
function hessianf{T<:Real}(x::T, y::T, z::T)
  w = zeros(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

args = [0.75, 2.5, -1.25]
output = f(FADHessian(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
