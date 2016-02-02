T = Float64
dummy_fsym = :sin
testexpr = :(sin(a) + exp(b) - tan(c) * cos(d))

testf = @eval (x::Vector) -> begin
    a,b,c,d = x
    return $testexpr
end

#############
# Gradients #
#############
N = 4
testx = grad_test_x(dummy_fsym, N)
testout = Array(T, N)
testresult = grad_test_result(testexpr, testx)

gradf! = forwarddiff_gradient!(testf, T)
gradf!(testx, testout)
@test_approx_eq testout testresult

gradf = forwarddiff_gradient(testf, T)
@test_approx_eq gradf(testx) testresult

#############
# Jacobians #
#############
N,M = 4,5
testx = jacob_test_x(dummy_fsym, N)
testout = Array(T, M, N)
testexpr_jac = [:(sin(a) + cos(b)), :(-tan(c)), :(4 * exp(d)), :(cos(b)^5), :(sin(a))]
testresult = jacob_test_result(testexpr_jac, testx)

jactestf = @eval (x::Vector) -> begin
    a,b,c,d = x
    return [$(testexpr_jac...)]
end

jacf! = forwarddiff_jacobian!(jactestf, T)
jacf!(testx, testout)
@test_approx_eq testout testresult

jacf = forwarddiff_jacobian(jactestf, T)
@test_approx_eq jacf(testx) testresult

############
# Hessians #
############
N = 6
testx = hess_test_x(dummy_fsym, N)
testout = Array(T, N, N)
testexpr_hess = :(sin(a) + exp(b) - tan(c) * cos(l) + sin(m) * exp(r))
testresult = hess_test_result(testexpr_hess, testx)

hess_testf = @eval (x::Vector) -> begin
    a,b,c,l,m,r = x
    return $testexpr_hess
end

hessf! = forwarddiff_hessian!(hess_testf, T)
hessf!(testx, testout)
@test_approx_eq testout testresult

hessf = forwarddiff_hessian(hess_testf, T)
@test_approx_eq hessf(testx) testresult

###########
# Tensors #
###########
N = 4
testx = tens_test_x(dummy_fsym, N)
testout = Array(T, N, N, N)
testresult = tens_test_result(testexpr, testx)

tensf! = forwarddiff_tensor!(testf, T)
tensf!(testx, testout)
@test_approx_eq testout testresult

tensf = forwarddiff_tensor(testf, T)
@test_approx_eq tensf(testx) testresult
