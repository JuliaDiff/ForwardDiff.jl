using ForwardDiff

println("Testing Partials...")
tic()
include("PartialsTest.jl")
println("done (took $(toq()) seconds).")

println("Testing Dual...")
tic()
include("DualTest.jl")
println("done (took $(toq()) seconds).")

println("Testing derivative functionality...")
tic()
include("DerivativeTest.jl")
println("done (took $(toq()) seconds).")

println("Testing gradient functionality...")
tic()
include("GradientTest.jl")
println("done (took $(toq()) seconds).")

println("Testing jacobian functionality...")
tic()
include("JacobianTest.jl")
println("done (took $(toq()) seconds).")

println("Testing hessian functionality...")
tic()
include("HessianTest.jl")
println("done (took $(toq()) seconds).")

println("Testing miscellaneous functionality...")
tic()
include("MiscTest.jl")
println("done (took $(toq()) seconds).")
