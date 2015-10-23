module ForwardDiffBenchmarks

using ForwardDiff
using DataFrames
using Benchmarks
using JLD

include("testfuncs.jl")

#########
# Types #
#########
immutable BenchmarkInfo
    time::Float64
    xlen::Int
    chunk_size::Int
end

immutable APIBenchmarkResults
    func_results::Vector{BenchmarkInfo}
    api_results::Dict{UTF8String,Matrix{BenchmarkInfo}}
end

# Steal DataFrames.jl's pretty-printing because I'm lazy
function toframe(arr::Array{BenchmarkInfo})
    times = [b.time for b in arr]
    xlens = [b.xlen for b in arr]
    chunk_sizes = [b.chunk_size for b in arr]
    return DataFrame(time=times, x_length=xlens, chunk_size=chunk_sizes)
end

function frameprintln(io::IO, arr::Array{BenchmarkInfo})
    df = toframe(arr)
    tmpio = IOBuffer()
    print(tmpio, df)
    s = takebuf_string(tmpio)
    println(io, s[(first(search(s, "\n"))+1):end])
end

function Base.show(io::IO, br::APIBenchmarkResults)
    println(io, "function times:")
    frameprintln(io, br.func_results)

    for (apifunc, infomat) in br.api_results
        println(io)
        println(io, "$apifunc times:")
        frameprintln(io, infomat)
    end
end

######################
# Running Benchmarks #
######################
const default_repeat = 5
const default_xlens = [16, 160, 1600]
const default_chunk_sizes = [ForwardDiff.default_chunk_size,1,4,8]

# will be replaced by Benchmarks.jl soon
function benchtime(f, x, repeat)
    min_time = Inf

    for i in 1:(repeat+1) # +1 for warm-up
        gc()
        this_time = @elapsed f(x)
        min_time = min(this_time, min_time)
    end

    return min_time
end

function run_func_benchmark(f, xs; repeat=default_repeat)
    results = Vector{BenchmarkInfo}(length(xs))
    for i in eachindex(xs)
        x = xs[i]
        t = benchtime(f, x, repeat)
        results[i] = BenchmarkInfo(t, length(x), -1)
    end
    return results
end

function run_api_benchmark(apifunc, f, xs;
                           repeat=default_repeat,
                           chunk_sizes=default_chunk_sizes)
    results = Matrix{BenchmarkInfo}(length(xs), length(chunk_sizes))
    for j in eachindex(chunk_sizes), i in eachindex(xs)
        x, c = xs[i], chunk_sizes[j]
        t = benchtime(apifunc(f, chunk_size=c), x, repeat)
        results[i,j] = BenchmarkInfo(t, length(x), c)
    end
    return results
end

const default_filepath = "forwarddiff_benchmarks.jld"

function run_all_benchmarks(fs...;
                            filepath=default_filepath,
                            repeat=default_repeat,
                            xlens=default_xlens,
                            chunk_sizes=default_chunk_sizes,
                            apifuncs=(ForwardDiff.gradient, ForwardDiff.hessian))
    results = Dict{UTF8String,APIBenchmarkResults}()
    xs = map(rand, xlens)
    jldopen(filepath, "w") do file
        for f in fs
            println("Performing all benchmarks for $f...")

            print("\tBenchmarking $f...")
            tic()
            func_results = run_func_benchmark(f, xs)
            println("done (took $(toq()) seconds).")

            api_results = Dict{UTF8String,Matrix{BenchmarkInfo}}()

            for apifunc in apifuncs
                print("\tBenchmarking $apifunc on $f...")
                tic()
                api_results[string(apifunc)] = run_api_benchmark(apifunc, f, xs;
                                                                 repeat=repeat,
                                                                 chunk_sizes=chunk_sizes)
                println("done (took $(toq()) seconds).")
            end

            result = APIBenchmarkResults(func_results, api_results)

            print("\tSaving results to $filepath...")
            tic()
            write(file, string(f), result)
            println("done (took $(toq()) seconds).")

            results[string(f)] = result
        end
    end
    return results
end

export run_all_benchmarks

end
