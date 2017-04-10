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

if Base.JLOptions().opt_level >= 3
    println("Testing SIMD vectorization...")
    tic()
    include("SIMDTest.jl")
    println("done (took $(toq()) seconds).")
end

# println("Testing deprecations...")
# tic()
# include("DeprecatedTest.jl")
# println("done (took $(toq()) seconds).")
