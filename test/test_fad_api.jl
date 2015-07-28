using Base.Test
using ForwardDiff
using Calculus

N = 4
P = Partials{N,Float64}
floatrange = 0.01:.01:.99
testx = rand(floatrange, N)

#########################
# Test Gradient methods #
#########################
# grad_testf: R⁴ -> R
function grad_testf(x::Vector)
    @assert length(x) == N
    return prod(x)
end

# hard code the correct jacobian for
# testf at the given vector x
function grad_test_result(x::Vector)
    @assert length(x) == N
    return [x[2]*x[3]*x[4],
            x[1]*x[3]*x[4],
            x[1]*x[2]*x[4],
            x[1]*x[2]*x[3]]
end

testout = Vector{Float64}(N)
testresult = grad_test_result(testx)

gradient!(grad_testf, testx, testout, P)
@test testout == testresult
fill!(testout, zero(eltype(testout)))

@test gradient(grad_testf, testx, P) == testresult

gradf! = gradient_func(grad_testf, P, mutates=true)
gradf!(testx, testout)
@test testout == testresult
fill!(testout, zero(eltype(testout)))

gradf = gradient_func(grad_testf, P, mutates=false)
@test gradf(testx) == testresult

#########################
# Test Jacobian methods #
#########################
# jac_testf: R⁴ -> R⁵
function jac_testf(x::Vector)
    @assert length(x) == N
    return [x[1], 
            5*x[3], 
            4*x[2]^2 - 2*x[3], 
            x[3]*sin(x[1]),
            sqrt(x[4])]
end

# hard code the correct jacobian for
# jac_testf at the given vector x
function jac_test_result(x::Vector)
    @assert length(x) == 4
    return [    1              0         0              0        ;
                0              0         5              0        ;
                0             8*x[2]    -2              0        ; 
            x[3]*cos(x[1])     0      sin(x[1])         0        ;
                0              0         0       1/(2*sqrt(x[4]))]
end


const M = 5

testout = Array(Float64, M, N)
testresult = jac_test_result(testx)

jacobian!(jac_testf, testx, testout, P)
@test testout == testresult
fill!(testout, zero(eltype(testout)))

@test jacobian(jac_testf, testx, P, M) == testresult

jacf! = jacobian_func(jac_testf, P, M, mutates=true)
jacf!(testx, testout)
@test testout == testresult
fill!(testout, zero(eltype(testout)))

jacf = jacobian_func(jac_testf, P, M, mutates=false)
@test jacf(testx) == testresult

#######################
# Test Tensor methods #
#######################
# tens_testf: R⁴ -> R
function tens_testf(x::Vector)
    @assert length(x) == N
    return prod(i->i^3, x)
end

function tens_deriv(i,j,k)
    wrt = [:a, :b, :c, :d]

    diff = differentiate(:(a^3 * b^3 * c^3 * d^3), wrt[k])
    diff = differentiate(diff, wrt[j])
    diff = differentiate(diff, wrt[i])

    str = string(diff)    
    str = replace(str, 'a', "x[1]")
    str = replace(str, 'b', "x[2]")
    str = replace(str, 'c', "x[3]")
    str = replace(str, 'd', "x[4]")

    return parse(str)
end

function tens_deriv(x::Vector, i, j, k)
    ex = tens_deriv(i, j, k)
    @eval begin
       x = $x
       return $ex
    end
end

# hard code the correct tensor for
# tens_testf at the given vector x
function tens_test_result(x::Vector)
    @assert length(x) == N
    return [tens_deriv(x, i, j, k) for i in 1:N, j in 1:N, k in 1:N]
end

testout = Array(Float64, N, N, N)
testresult = tens_test_result(testx)

tensor!(tens_testf, testx, testout, P)
@test testout == testresult
fill!(testout, zero(eltype(testout)))

@test tensor(tens_testf, testx, P) == testresult

tensf! = tensor_func(tens_testf, P, mutates=true)
tensf!(testx, testout)
@test testout == testresult
fill!(testout, zero(eltype(testout)))

tensf = tensor_func(tens_testf, P, mutates=false)
@test tensf(testx) == testresult
