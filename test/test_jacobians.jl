using Base.Test
using ForwardDiff
using Calculus

N = 4
M = 5

testout = Array(Float64, M, N)

function jacob_deriv_ij(fs::Vector{Expr}, x::Vector, i, j)
    var_syms = [:a, :b, :c, :d]
    diff_expr = differentiate(fs[i], var_syms[j])
    @eval begin
        a,b,c,d = $x
        return $diff_expr
    end
end

function jacob_test_result(fs::Vector{Expr}, x::Vector)
    return [jacob_deriv_ij(fs, x, i, j) for i in 1:M, j in 1:N]
end

function jacob_test_x(fsym, N)
    randrange = 0.01:.01:.99
    needs_rand_mod = tuple(:acosh, :acoth, :asec, :acsc, :asecd, :acscd)

    if fsym in needs_rand_mod
        randrange += 1
    end

    return rand(randrange, N)
end

chunk_sizes = (0, 1, Int(N/2), N)

for fsym in ForwardDiff.fad_supported_univar_funcs
    testexprs = [:($(fsym)(a) + $(fsym)(b)),
                :(- $(fsym)(c)),
                :(4 * $(fsym)(d)),
                :($(fsym)(b)^5),
                :($(fsym)(a))]

    @eval function testf(x::Vector) 
        a,b,c,d = x
        return [$(testexprs...)]
    end

    for chunk in chunk_sizes
        try
            testx = jacob_test_x(fsym, N)
            testresult = jacob_test_result(testexprs, testx)
            ForwardDiff.jacobian!(testout, testf, testx, chunk_size=chunk)
            @test_approx_eq testout testresult

            @test_approx_eq ForwardDiff.jacobian(testf, testx, chunk_size=chunk) testresult

            jacf! = ForwardDiff.jacobian(testf, mutates=true)
            jacf!(testout, testx, chunk_size=chunk)
            @test_approx_eq testout testresult

            jacf = ForwardDiff.jacobian(testf, mutates=false)
            @test_approx_eq jacf(testx, chunk_size=chunk) testresult
        catch err
            warn("Failure when testing Jacobians involving $fsym with chunk_size=$chunk:")
            throw(err)
        end
    end
end