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
  w = Array(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

d2fdxxx(x, y, z) = 24*(5*x^3+x*y*z)
d2fdxyx(x, y, z) = 12*(y^2+x^2*z)
d2fdxyy(x, y, z) = 6*y*(4*x-3*z^2)
d2fdxzx(x, y, z) = 12*x^2*y
d2fdxzy(x, y, z) = -2+4*x^3-18*y^2*z
d2fdxzz(x, y, z) = -6*y^3
d2fdyxx(x, y, z) = 12*(y^2+x^2*z)
d2fdyyx(x, y, z) = 6*y*(4*x-3*z^2)
d2fdyyy(x, y, z) = 6*(2*x^2-10*y^2-3*x*z^2)
d2fdyzx(x, y, z) = -2+4*x^3-18*y^2*z
d2fdyzy(x, y, z) = -4*(2+9*x*y*z)
d2fdyzz(x, y, z) = 2-18*x*y^2
d2fdzxx(x, y, z) = 12*x^2*y
d2fdzyx(x, y, z) = -2+4*x^3-18*y^2*z
d2fdzyy(x, y, z) = -4*(2+9*x*y*z)
d2fdzzx(x, y, z) = -6*y^3
d2fdzzy(x, y, z) = 2-18*x*y^2
d2fdzzz(x, y, z) = 0
function tensorf{T<:Real}(x::T, y::T, z::T)
  w = Array(T, 3, 3, 3)
  w[1, 1, 1] = d2fdxxx(x, y, z)
  w[2, 1, 1] = w[1, 2, 1] = d2fdxyx(x, y, z)
  w[2, 2, 1] = d2fdxyy(x, y, z)
  w[3, 1, 1] = w[1, 3, 1] = d2fdxzx(x, y, z)
  w[3, 2, 1] = w[2, 3, 1] = d2fdxzy(x, y, z)
  w[3, 3, 1] = d2fdxzz(x, y, z)
  w[1, 1, 2] = d2fdyxx(x, y, z)
  w[2, 1, 2] = w[1, 2, 2] = d2fdyyx(x, y, z)
  w[2, 2, 2] = d2fdyyy(x, y, z)
  w[3, 1, 2] = w[1, 3, 2] = d2fdyzx(x, y, z)
  w[3, 2, 2] = w[2, 3, 2] = d2fdyzy(x, y, z)
  w[3, 3, 2] = d2fdyzz(x, y, z)
  w[1, 1, 3] = d2fdzxx(x, y, z)
  w[2, 1, 3] = w[1, 2, 3] = d2fdzyx(x, y, z)
  w[2, 2, 3] = d2fdzyy(x, y, z)
  w[3, 1, 3] = w[1, 3, 3] = d2fdzzx(x, y, z)
  w[3, 2, 3] = w[2, 3, 3] = d2fdzzy(x, y, z)
  w[3, 3, 3] = d2fdzzz(x, y, z)
  w
end

args = [2.3, -1.5, -4.]
output = f(FADTensor(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
@test_approx_eq tensor(output) tensorf(args...)

# Testing the API

g = forwarddiff_tensor(f, Float64, fadtype=:typed)
output = g(args)
@test_approx_eq output tensorf(args...)

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
  w = Array(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

d2fdxxx(x, y, z) = 60*x^2*z/y^4
d2fdxyx(x, y, z) = -80*x^3*z/y^5
d2fdxyy(x, y, z) = 100*x^4*z/y^6
d2fdxzx(x, y, z) = 20*x^3/y^4
d2fdxzy(x, y, z) = -20*x^4/y^5
d2fdxzz(x, y, z) = 0
d2fdyxx(x, y, z) = -80*x^3*z/y^5
d2fdyyx(x, y, z) = 100*x^4*z/y^6
d2fdyyy(x, y, z) = -120*x^5*z/y^7
d2fdyzx(x, y, z) = -20*x^4/y^5
d2fdyzy(x, y, z) = 20*x^5/y^6
d2fdyzz(x, y, z) = 0
d2fdzxx(x, y, z) = 20*x^3/y^4
d2fdzyx(x, y, z) = -20*x^4/y^5
d2fdzyy(x, y, z) = 20*x^5/y^6
d2fdzzx(x, y, z) = 0
d2fdzzy(x, y, z) = 0
d2fdzzz(x, y, z) = 6/z^4
function tensorf{T<:Real}(x::T, y::T, z::T)
  w = Array(T, 3, 3, 3)
  w[1, 1, 1] = d2fdxxx(x, y, z)
  w[2, 1, 1] = w[1, 2, 1] = d2fdxyx(x, y, z)
  w[2, 2, 1] = d2fdxyy(x, y, z)
  w[3, 1, 1] = w[1, 3, 1] = d2fdxzx(x, y, z)
  w[3, 2, 1] = w[2, 3, 1] = d2fdxzy(x, y, z)
  w[3, 3, 1] = d2fdxzz(x, y, z)
  w[1, 1, 2] = d2fdyxx(x, y, z)
  w[2, 1, 2] = w[1, 2, 2] = d2fdyyx(x, y, z)
  w[2, 2, 2] = d2fdyyy(x, y, z)
  w[3, 1, 2] = w[1, 3, 2] = d2fdyzx(x, y, z)
  w[3, 2, 2] = w[2, 3, 2] = d2fdyzy(x, y, z)
  w[3, 3, 2] = d2fdyzz(x, y, z)
  w[1, 1, 3] = d2fdzxx(x, y, z)
  w[2, 1, 3] = w[1, 2, 3] = d2fdzyx(x, y, z)
  w[2, 2, 3] = d2fdzyy(x, y, z)
  w[3, 1, 3] = w[1, 3, 3] = d2fdzzx(x, y, z)
  w[3, 2, 3] = w[2, 3, 3] = d2fdzzy(x, y, z)
  w[3, 3, 3] = d2fdzzz(x, y, z)
  w
end

args = [2., 5., -1.]
output = f(FADTensor(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
@test_approx_eq tensor(output) tensorf(args...)

# Testing square roots, qubic roots and rational powers

f(x) = sqrt(x)
gradf(x) = 1/(2*sqrt(x))
hessianf(x) = -1/(4*x^1.5)
tensorf(x) = 3/(8*x^2.5)

args = [1.17]
output = f(FADTensor(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
@test_approx_eq tensor(output) tensorf(args...)

f(x) = cbrt(x)
gradf(x) = 1/(3*x^(2//3))
hessianf(x) = -2/(9*x^(5//3))
tensorf(x) = 10/(27*x^(8//3))

args = [1.61]
output = f(FADTensor(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
@test_approx_eq tensor(output) tensorf(args...)

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
  w = Array(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

d2fdxxx(x, y, z) = -440/(27*x^(14/3))+3*z^4/(8*x^2.5*cbrt(y))
d2fdxyx(x, y, z) = z^4/(12*x^1.5*y*cbrt(y))
d2fdxyy(x, y, z) = 2*z^4/(9*sqrt(x)*y^2*cbrt(y))
d2fdxzx(x, y, z) = -z^3/(x^1.5*cbrt(y))
d2fdxzy(x, y, z) = -2*z^3/(3*sqrt(x)*y*cbrt(y))
d2fdxzz(x, y, z) = 6*z^2/(sqrt(x)*cbrt(y))
d2fdyxx(x, y, z) = z^4/(12*x^1.5*y*cbrt(y))
d2fdyyx(x, y, z) = 2*z^4/(9*sqrt(x)*y^2*cbrt(y))
d2fdyyy(x, y, z) = -28*sqrt(x)*z^4/(27*y^3*cbrt(y))
d2fdyzx(x, y, z) = -2*z^3/(3*sqrt(x)*y*cbrt(y))
d2fdyzy(x, y, z) = 16*sqrt(x)*z^3/(9*y^2*cbrt(y))
d2fdyzz(x, y, z) = -4*sqrt(x)*z^2/(y*cbrt(y))
d2fdzxx(x, y, z) = -z^3/(x^1.5*cbrt(y))
d2fdzyx(x, y, z) = -2*z^3/(3*sqrt(x)*y*cbrt(y))
d2fdzyy(x, y, z) = 16*sqrt(x)*z^3/(9*y^2*cbrt(y))
d2fdzzx(x, y, z) = 6*z^2/(sqrt(x)*cbrt(y))
d2fdzzy(x, y, z) = -4*sqrt(x)*z^2/(y*cbrt(y))
d2fdzzz(x, y, z) = 24*sqrt(x)*z/cbrt(y)
function tensorf{T<:Real}(x::T, y::T, z::T)
  w = Array(T, 3, 3, 3)
  w[1, 1, 1] = d2fdxxx(x, y, z)
  w[2, 1, 1] = w[1, 2, 1] = d2fdxyx(x, y, z)
  w[2, 2, 1] = d2fdxyy(x, y, z)
  w[3, 1, 1] = w[1, 3, 1] = d2fdxzx(x, y, z)
  w[3, 2, 1] = w[2, 3, 1] = d2fdxzy(x, y, z)
  w[3, 3, 1] = d2fdxzz(x, y, z)
  w[1, 1, 2] = d2fdyxx(x, y, z)
  w[2, 1, 2] = w[1, 2, 2] = d2fdyyx(x, y, z)
  w[2, 2, 2] = d2fdyyy(x, y, z)
  w[3, 1, 2] = w[1, 3, 2] = d2fdyzx(x, y, z)
  w[3, 2, 2] = w[2, 3, 2] = d2fdyzy(x, y, z)
  w[3, 3, 2] = d2fdyzz(x, y, z)
  w[1, 1, 3] = d2fdzxx(x, y, z)
  w[2, 1, 3] = w[1, 2, 3] = d2fdzyx(x, y, z)
  w[2, 2, 3] = d2fdzyy(x, y, z)
  w[3, 1, 3] = w[1, 3, 3] = d2fdzzx(x, y, z)
  w[3, 2, 3] = w[2, 3, 3] = d2fdzzy(x, y, z)
  w[3, 3, 3] = d2fdzzz(x, y, z)
  w
end

args = [0.75, 2.5, -1.25]
output = f(FADTensor(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
@test_approx_eq tensor(output) tensorf(args...)

# Testing exp, log, log2 and and log10 

f(x, y, z) = log2(x)*exp(y*z)+log(x^4*y)/log10(z)

dfdx(x, y, z) = (exp(y*z)/log(2)+4*log(10)/log(z))/x
dfdy(x, y, z) = exp(y*z)*z*log(x)/log(2)+log(10)/(y*log(z))
dfdz(x, y, z) = exp(y*z)*y*log(x)/log(2)-log(10)*log(x^4*y)/(z*log(z)^2)
gradf(x, y, z) = [dfdx(x, y, z), dfdy(x, y, z), dfdz(x, y, z)]

dfdxx(x, y, z) = -(exp(y*z)/log(2)+4*log(10)/log(z))/x^2
dfdxy(x, y, z) = exp(y*z)*z/(x*log(2))
dfdyy(x, y, z) = exp(y*z)*z^2*log(x)/log(2)-log(10)/(y^2*log(z))
dfdxz(x, y, z) = (exp(y*z)*y/log(2)-4*log(10)/(z*log(z)^2))/x
dfdyz(x, y, z) = exp(y*z)*(1+y*z)*log(x)/log(2)-log(10)/(y*z*log(z)^2)
dfdzz(x, y, z) = exp(y*z)*y^2*log(x)/log(2)+log(10)*log(x^4*y)*(2+log(z))/(z^2*log(z)^3)
function hessianf{T<:Real}(x::T, y::T, z::T)
  w = Array(T, 3, 3)
  w[1, 1] = dfdxx(x, y, z)
  w[2, 1] = w[1, 2] = dfdxy(x, y, z)
  w[2, 2] = dfdyy(x, y, z)
  w[3, 1] = w[1, 3]= dfdxz(x, y, z)
  w[3, 2] = w[2, 3] = dfdyz(x, y, z)
  w[3, 3] = dfdzz(x, y, z)
  w
end

d2fdxxx(x, y, z) = 2*(exp(y*z)/log(2)+4*log(10)/log(z))/x^3
d2fdxyx(x, y, z) = -z*exp(y*z)/(log(2)*x^2)
d2fdxyy(x, y, z) = z^2*exp(y*z)/(log(2)*x)
d2fdxzx(x, y, z) = (-y*exp(y*z)/log(2)+4*log(10)/(z*log(z)^2))/x^2
d2fdxzy(x, y, z) = exp(y*z)*(1+y*z)/(log(2)*x)
d2fdxzz(x, y, z) = (y^2*exp(y*z)/log(2)+4*log(10)*(2+log(z))/(z^2*log(z)^3))/x
d2fdyxx(x, y, z) = -z*exp(y*z)/(log(2)*x^2)
d2fdyyx(x, y, z) = z^2*exp(y*z)/(log(2)*x)
d2fdyyy(x, y, z) = z^3*exp(y*z)*log(x)/log(2)+log(100)/(y^3*log(z))
d2fdyzx(x, y, z) = exp(y*z)*(1+y*z)/(log(2)*x)
d2fdyzy(x, y, z) = z*(2+y*z)*exp(y*z)*log(x)/log(2)+log(10)/(y^2*z*log(z)^2)
d2fdyzz(x, y, z) = y*(2+y*z)*exp(y*z)*log(x)/log(2)+log(10)*(2+log(z))/(y*z^2*log(z)^3)
d2fdzxx(x, y, z) = (-y*exp(y*z)/log(2)+4*log(10)/(z*log(z)^2))/x^2
d2fdzyx(x, y, z) = exp(y*z)*(1+y*z)/(log(2)*x)
d2fdzyy(x, y, z) = z*(2+y*z)*exp(y*z)*log(x)/log(2)+log(10)/(y^2*z*log(z)^2)
d2fdzzx(x, y, z) = (y^2*exp(y*z)/log(2)+4*log(10)*(2+log(z))/(z^2*log(z)^3))/x
d2fdzzy(x, y, z) = y*(2+y*z)*exp(y*z)*log(x)/log(2)+log(10)*(2+log(z))/(y*z^2*log(z)^3)
d2fdzzz(x, y, z) = y^3*exp(y*z)*log(x)/log(2)-2*log(10)*log(x^4*y)*(3+log(z)*(3+log(z)))/(z^3*log(z)^4)
function tensorf{T<:Real}(x::T, y::T, z::T)
  w = Array(T, 3, 3, 3)
  w[1, 1, 1] = d2fdxxx(x, y, z)
  w[2, 1, 1] = w[1, 2, 1] = d2fdxyx(x, y, z)
  w[2, 2, 1] = d2fdxyy(x, y, z)
  w[3, 1, 1] = w[1, 3, 1] = d2fdxzx(x, y, z)
  w[3, 2, 1] = w[2, 3, 1] = d2fdxzy(x, y, z)
  w[3, 3, 1] = d2fdxzz(x, y, z)
  w[1, 1, 2] = d2fdyxx(x, y, z)
  w[2, 1, 2] = w[1, 2, 2] = d2fdyyx(x, y, z)
  w[2, 2, 2] = d2fdyyy(x, y, z)
  w[3, 1, 2] = w[1, 3, 2] = d2fdyzx(x, y, z)
  w[3, 2, 2] = w[2, 3, 2] = d2fdyzy(x, y, z)
  w[3, 3, 2] = d2fdyzz(x, y, z)
  w[1, 1, 3] = d2fdzxx(x, y, z)
  w[2, 1, 3] = w[1, 2, 3] = d2fdzyx(x, y, z)
  w[2, 2, 3] = d2fdzyy(x, y, z)
  w[3, 1, 3] = w[1, 3, 3] = d2fdzzx(x, y, z)
  w[3, 2, 3] = w[2, 3, 3] = d2fdzzy(x, y, z)
  w[3, 3, 3] = d2fdzzz(x, y, z)
  w
end

args = [0.3, 1.1, 2.25]
output = f(FADTensor(args)...)

@test_approx_eq value(output) f(args...)
@test_approx_eq grad(output) gradf(args...)
@test_approx_eq hessian(output) hessianf(args...)
@test_approx_eq tensor(output) tensorf(args...)
