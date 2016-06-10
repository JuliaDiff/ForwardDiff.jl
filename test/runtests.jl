using ForwardDiff

print("Testing Partials...")
tic()
include("PartialsTest.jl")
println("done (took $(toq()) seconds).")

print("Testing Dual...")
tic()
include("DualTest.jl")
println("done (took $(toq()) seconds).")

print("Testing gradient-related functionality...")
tic()
include("GradientTest.jl")
println("done (took $(toq()) seconds).")

print("Testing jacobian-related functionality...")
tic()
include("JacobianTest.jl")
println("done (took $(toq()) seconds).")
