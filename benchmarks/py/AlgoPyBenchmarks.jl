module AlgoPyBenchmarks

    using DataFrame
    using PyCall

    const algopy_bench_file = joinpath(Pkg.dir("ForwardDiff"), "benchmarks", "py", "benchmarks.py")

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

end
