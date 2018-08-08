using ForwardDiff

println("Testing Partials...")
t = @elapsed include("PartialsTest.jl")
println("done (took $t seconds).")

println("Testing Dual...")
t = @elapsed include("DualTest.jl")
println("done (took $t seconds).")

println("Testing derivative functionality...")
t = @elapsed include("DerivativeTest.jl")
println("done (took $t seconds).")

println("Testing gradient functionality...")
t = @elapsed include("GradientTest.jl")
println("done (took $t seconds).")

println("Testing jacobian functionality...")
t = @elapsed include("JacobianTest.jl")
println("done (took $t seconds).")

println("Testing hessian functionality...")
t = @elapsed include("HessianTest.jl")
println("done (took $t seconds).")

println("Testing perturbation confusion functionality...")
t = @elapsed include("ConfusionTest.jl")
println("done (took $t seconds).")

println("Testing miscellaneous functionality...")
t = @elapsed include("MiscTest.jl")
println("done (took $t seconds).")

if Base.JLOptions().opt_level >= 3
    println("Testing SIMD vectorization...")
    t = @elapsed include("SIMDTest.jl")
    println("done (took $t seconds).")
end
