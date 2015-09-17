using ForwardDiff
using DataFrames
using JLD

##################
# Test functions #
##################
function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

function ackley(x)
    a, b, c = 20.0, -0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += i^2
    end
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + e)
end

self_weighted_logit(x) = inv(1.0 + exp(-dot(x, x)))

#############################
# Benchmark utility methods #
#############################
const folder_path = joinpath(Pkg.dir("ForwardDiff"), "benchmarks")
const data_path = joinpath(folder_path, "benchmark_data")

data_name(f) = "$(f)_benchmarks"
data_file(f) = joinpath(data_path, "$(data_name(f)).jld")

function bench_func(f::Function, x, repeat)

    min_time = Inf

    for i in 1:(repeat+1) # +1 for warm-up
        gc()
        this_time = @elapsed f(x)
        min_time = min(this_time, min_time)
    end

    return min_time
end

function run_benchmark(f;
                       repeat=5,
                       xlens=(16,1600,16000),
                       chunk_sizes=(ForwardDiff.default_chunk_size,1,2,4,8,16))

    benchdf = DataFrame(time=Float64[],
                        func=Char[],
                        xlen=Int[],
                        chunk_size=Int[])

    for xlen in xlens
        x = rand(xlen)
        push!(benchdf, [bench_func(f, x, repeat), 'f', xlen, -1])
        for c in chunk_sizes
            g = ForwardDiff.gradient(f, chunk_size=c)
            push!(benchdf, [bench_func(g, x, repeat), 'g', xlen, c])
        end
    end

    return benchdf
end

function run_benchmarks(fs...)
    for f in fs
        print("Performing default benchmarks for $f...")
        tic()
        result = run_benchmark(f)
        println("done (took $(toq()) seconds).")
        file = data_file(f)
        print("\tSaving data to $(file)...")
        save(file, data_name(f), result)
        println("done.")
    end
end

get_benchmark(f) = first(values(load(data_file(f))))
get_benchmarks(fs...) = [symbol(f) => get_benchmark(f) for f in fs]

##################
# Run Benchmarks #
##################
const default_fs = (ackley, rosenbrock, self_weighted_logit)

run_default_benchmarks() = run_benchmarks(default_fs...)

#######################
# Retrieve benchmarks #
#######################
using PyCall

# allow loading of Python modules from current directory
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport benchmarks

# Grab ForwardDiff benchmarks from benchmark_data
get_fordiff_benchmarks() = get_benchmarks(default_fs...)

# Grab AlgoPy benchmarks from benchmark_data
function algopy_bench_to_df(bench::Dict)
    bench_dfs = Dict{Symbol, DataFrame}()
    for (k,v) in bench
        bench_dfs[symbol(k)] = DataFrame(v)
    end
    return bench_dfs
end

get_algopy_benchmarks() = algopy_bench_to_df(benchmarks.get_default_benchmarks())
