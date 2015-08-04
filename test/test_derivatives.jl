using Base.Test
using ForwardDiff
using Calculus

###########################
# Test taking derivatives #
###########################
N = 4

function deriv_test_x(fsym)
    randrange = 0.01:.01:.99
    needs_rand_mod = tuple(acosh, acoth, asec, acsc, asecd, acscd)

    if fsym in needs_rand_mod
        randrange += 1
    end

    return rand(randrange)
end

function test_approx_deriv(a::Vector, b::Vector)
    @assert length(a) == length(b)
    for i in eachindex(a)
        @test_approx_eq a[i] b[i]
    end
end

for fsym in ForwardDiff.fad_supported_univar_funcs
    deriv = Calculus.differentiate(:($(fsym)(x)))
    try 
        @eval begin 
            x = deriv_test_x($fsym)
            f_num = $fsym
            result = $deriv
            
            @test_approx_eq result ForwardDiff.derivative(f_num, x)
            @test_approx_eq result ForwardDiff.derivative(f_num)(x)

            f_vec = t -> [f_num(t) for i=1:N]
            result_vec = [result for i=1:N]

            test_approx_deriv(result_vec, ForwardDiff.derivative(f_vec, x))
            test_approx_deriv(result_vec, ForwardDiff.derivative(f_vec)(x))
            test_approx_deriv(result_vec, ForwardDiff.derivative!(f_vec, x, Vector{Float64}(N)))
        end
    catch err
        error("Failure when testing derivative of $fsym: $err")
    end
end
