using ForwardDiff
using BenchmarkTrackers

@tracker TRACKER

samerand(args...) = rand(MersenneTwister(1), args...)

include("../test/TestFuncs.jl")
include("DerivativeBenchmark.jl")
include("GradientBenchmark.jl")
include("JacobianBenchmark.jl")

results = run(TRACKER)
