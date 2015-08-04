using Base.Test
using ForwardDiff
using Calculus

############################
# Test taking the Jacobian #
############################

N = 4
floatrange = 0.01:.01:.99
testx = rand(floatrange, N)

# jac_testf: R⁴ -> R⁵
function jac_testf(x::Vector)
    @assert length(x) == N
    return [x[1], 
            5*x[3], 
            4*x[2]^2 - 2*x[3], 
            x[3]*sin(x[1]),
            sqrt(x[4])]
end

# hard code the correct ForwardDiff.jacobian for
# jac_testf at the given vector x
function jac_test_result(x::Vector)
    @assert length(x) == 4
    return [    1              0         0              0        ;
                0              0         5              0        ;
                0             8*x[2]    -2              0        ; 
            x[3]*cos(x[1])     0      sin(x[1])         0        ;
                0              0         0       1/(2*sqrt(x[4]))]
end

testout = Array(Float64, 5, N)
testresult = jac_test_result(testx)

ForwardDiff.jacobian!(jac_testf, testx, testout)
@test testout == testresult

@test ForwardDiff.jacobian(jac_testf, testx) == testresult

jacf! = ForwardDiff.jacobian(jac_testf, mutates=true)
jacf!(testx, testout)
@test testout == testresult

jacf = ForwardDiff.jacobian(jac_testf, mutates=false)
@test jacf(testx) == testresult