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

chunk_sizes = (ForwardDiff.default_chunk_size, 1, Int(N/2), N)

for fsym in ForwardDiff.auto_defined_unary_funcs
    testexprs = [:($(fsym)(a) + $(fsym)(b)),
                :(- $(fsym)(c)),
                :(4 * $(fsym)(d)),
                :($(fsym)(b)^5),
                :($(fsym)(a))]

    testf = @eval (x::Vector) -> begin
        a,b,c,d = x
        return [$(testexprs...)]
    end

    testf! = @eval (output::Vector, x::Vector) -> begin
        a,b,c,d = x
        output[1] = $(testexprs[1])
        output[2] = $(testexprs[2])
        output[3] = $(testexprs[3])
        output[4] = $(testexprs[4])
        output[5] = $(testexprs[5])
    end

    for chunk in chunk_sizes
        try
            testx = jacob_test_x(fsym, N)
            val_result = testf(testx)
            jacob_result = jacob_test_result(testexprs, testx)

            # Non-AllResults
            test_jacob = (testout) -> @test_approx_eq testout jacob_result

            ForwardDiff.jacobian!(testout, testf, testx; chunk_size=chunk)
            test_jacob(testout)

            test_jacob(ForwardDiff.jacobian(testf, testx; chunk_size=chunk))

            jacf! = ForwardDiff.jacobian(testf; mutates=true, chunk_size=chunk)
            testout = similar(testout)
            jacf!(testout, testx)
            test_jacob(testout)

            jacf = ForwardDiff.jacobian(testf; mutates=false, chunk_size=chunk)
            test_jacob(jacf(testx))

            jacf! = ForwardDiff.jacobian(testf!; mutates=true, chunk_size=chunk, output_length=M)
            testout = similar(testout)
            jacf!(testout, testx)
            test_jacob(testout)

            jacf = ForwardDiff.jacobian(testf!; mutates=false, chunk_size=chunk, output_length=M)
            test_jacob(jacf(testx))

            # AllResults
            test_all_results = (testout, results) -> begin
                @test_approx_eq ForwardDiff.value(results) val_result
                test_jacob(ForwardDiff.jacobian(results))
                test_jacob(testout)
            end

            testout = similar(testout)
            results = ForwardDiff.jacobian!(testout, testf, testx, AllResults; chunk_size=chunk)
            test_all_results(testout, results[2])

            testout = similar(testout)
            testout, results2 = ForwardDiff.jacobian(testf, testx, AllResults; chunk_size=chunk)
            test_all_results(testout, results2)

            jacf! = ForwardDiff.jacobian(testf, AllResults; mutates=true, chunk_size=chunk)
            testout = similar(testout)
            results3 = jacf!(testout, testx)
            test_all_results(testout, results3[2])

            jacf = ForwardDiff.jacobian(testf, AllResults; mutates=false, chunk_size=chunk)
            testout = similar(testout)
            testout, results4 = jacf(testx)
            test_all_results(testout, results4)

            jacf! = ForwardDiff.jacobian(testf!, AllResults; mutates=true, chunk_size=chunk, output_length=M)
            testout = similar(testout)
            results5 = jacf!(testout, testx)
            test_all_results(testout, results5[2])

            jacf = ForwardDiff.jacobian(testf!, AllResults; mutates=false, chunk_size=chunk, output_length=M)
            testout = similar(testout)
            testout, results6 = jacf(testx)
            test_all_results(testout, results6)
        catch err
            warn("Failure when testing Jacobians involving $fsym with chunk_size=$chunk:")
            throw(err)
        end
    end
end
