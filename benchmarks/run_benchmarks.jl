println("Running Julia benchmarks...")
include("benchmarks.jl")
run_default_benchmarks()
println("Done with Julia benchmarks.")

println("Running Python benchmarks...")
path = joinpath(Pkg.dir("ForwardDiff"), "benchmarks", "benchmarks.py")
run(`python $path`)
println("Done with Python benchmarks.")

