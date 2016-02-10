using ForwardDiff
using BenchmarkTrackers

@tracker TRACKER

samerand(args...) = rand(MersenneTwister(1), args...)

include("testfuncs.jl")
include("DerivativeBenchmark.jl")
include("GradientBenchmark.jl")

results = run(TRACKER)
