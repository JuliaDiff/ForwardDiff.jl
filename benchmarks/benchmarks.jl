using ForwardDiff
using DataFrames
using JLD

##################
# Test functions #
##################
sqr(i) = i*i

function ackley(x)
    a, b, c = 20.0, -0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += sqr(i)
    end
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + e)
end

function rosenbrock(x)
    a, b = 100.0, 1.0
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += sqr(b - x[i]) + a*sqr(x[i+1] - sqr(x[i]))
    end
    return result
end

self_weighted_logit(x) = inv(1.0 + exp(-dot(x, x)))

#############################
# Benchmark utility methods #
#############################
function bench_func(f_expr::Expr, x, repeat)

    @eval function test()
        x = $x
        return $f_expr
    end

    min_time = Inf

    for i in 1:(repeat+1) # +1 for warm-up
        gc()
        this_time = @elapsed test()
        min_time = min(this_time, min_time)
    end

    return min_time
end

function bench_fad(f;
                   repeat=5,
                   xlens=(16,160),#1600,16000),
                   chunk_sizes=(ForwardDiff.default_chunk_size,1,2,4,8,16))

    benchdf = DataFrame(time=Float64[],
                        func=Char[],
                        xlen=Int[],
                        chunk_size=Int[])
    f_expr = :(($f)(x))
    g = ForwardDiff.gradient(f)

    for xlen in xlens
        x = rand(xlen)
        push!(benchdf, [bench_func(f_expr, x, repeat), 'f', xlen, -1])
        for c in chunk_sizes
            g_expr = :(($g)(x, chunk_size=$c))
            push!(benchdf, [bench_func(g_expr, x, repeat), 'g', xlen, c])
        end
    end

    return benchdf
end

function default_benchmark(fs...)
    folder_path = joinpath(Pkg.dir("ForwardDiff"), "benchmarks", "benchmark_data")
    for f in fs
        print("Performing default benchmarks for $f...")
        tic()
        result = bench_fad(f)
        println("done (took $(toq()) seconds).")
        file_path =  joinpath(folder_path, "$(f)_times.jld")
        print("\tSaving data to $(file_path)...")
        save(file_path, "$(f)_times", result)
        println("done.")
    end
    println("Done with all benchmarks!")
end

##################
# Run Benchmarks #
##################
main() = default_benchmark(ackley, rosenbrock, self_weighted_logit)
