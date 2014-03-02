using ForwardDiff
using Base.Test


f(x) = exp(sin(x[1]*x[2]))
g! = forwarddiff_gradient(f, Float64, 2)
out = zeros(2)
xvals = [3.4, 2.1]
g!(xvals, out)
q = xvals[1]*xvals[2]
@test_approx_eq f(xvals) exp(sin(q))
@test_approx_eq out[1] xvals[2]*cos(q)*exp(sin(q))
@test_approx_eq out[2] xvals[1]*cos(q)*exp(sin(q))

f(x) = exp(sin(x[1]*x[2]+x[3]^2)) + 2x[1]*x[1]
g! = forwarddiff_gradient(f, Float64, 3)
xvals = [3.4,2.1,6.7]
fval = f(xvals)
out = zeros(3)
g!(xvals, out)
q = xvals[1]*xvals[2]+xvals[3]^2
@test_approx_eq fval exp(sin(q)) + 2xvals[1]^2
@test_approx_eq out[1] xvals[2]*cos(q)*exp(sin(q)) + 4xvals[1]
@test_approx_eq out[2] xvals[1]*cos(q)*exp(sin(q))
@test_approx_eq out[3] 2xvals[3]*cos(q)*exp(sin(q))


# from NLSolve.jl README
function f!(x, fvec)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end

function jcorrect!(x, fjac)
    fjac[1, 1] = x[2]^3-7
    fjac[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    fjac[2, 1] = x[2]*u
    fjac[2, 2] = u
end

xvals = [5.6,8.2]
j! = forwarddiff_jacobian(f!, Float64, 2, 2)
out = zeros(2,2)
jout = zeros(2,2)
jcorrect!(xvals, jout)
j!(xvals, out)
@test_approx_eq out jout



