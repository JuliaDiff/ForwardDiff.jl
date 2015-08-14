using Base.Test
using ForwardDiff
using Calculus

###########################
# Test taking derivatives #
###########################
N = 4
M = 2N
L = 3N

testout = Array(Float64, N, M, L)

function deriv_test_x(fsym)
    randrange = 0.01:.01:.99
    needs_rand_mod = tuple(acosh, acoth, asec, acsc, asecd, acscd)

    if fsym in needs_rand_mod
        randrange += 1
    end

    return rand(randrange)
end

function test_approx_deriv(a, b)
    @assert length(a) == length(b)
    for i in eachindex(a)
        @test_approx_eq a[i] b[i]
    end
end

for fsym in ForwardDiff.fad_supported_univar_funcs
    func_expr = :($(fsym)(x) + 4^$(fsym)(x) - x * $(fsym)(x))
    deriv = Calculus.differentiate(func_expr)
    try 
        @eval begin 
            x = deriv_test_x($fsym)
            testdf = x -> $func_expr
            result = $deriv
            
            @test_approx_eq result ForwardDiff.derivative(testdf, x)
            @test_approx_eq result ForwardDiff.derivative(testdf)(x)

            testdf_arr = t -> [testdf(t) for i in 1:N, j in 1:M, k in 1:L]
            result_arr = [result for i in 1:N, j in 1:M, k in 1:L]

            test_approx_deriv(result_arr, ForwardDiff.derivative(testdf_arr, x))
            test_approx_deriv(result_arr, ForwardDiff.derivative(testdf_arr)(x))
            test_approx_deriv(result_arr, ForwardDiff.derivative!(testout, testdf_arr, x))
        end
    catch err
        error("Failure when testing derivative of $fsym: $err")
    end
end
