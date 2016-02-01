using ForwardDiff

print("Testing Partials...")
tic()
include("PartialsTest.jl")
println("done (took $(toq()) seconds).")

print("Testing DiffNumber...")
tic()
include("DiffNumberTest.jl")
println("done (took $(toq()) seconds).")

print("Testing Gradient-related functionality...")
tic()
include("GradientTest.jl")
println("done (took $(toq()) seconds).")
