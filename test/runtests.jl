using ForwardDiff

print("Testing GradientNumbers and ForwardDiff.gradient...")
tic()
include("test_gradients.jl")
println("done (took $(toq()) seconds).")

print("Testing HessianNumbers and ForwardDiff.hessian...")
tic()
include("test_hessians.jl")
println("done (took $(toq()) seconds).")

print("Testing TensorNumbers and ForwardDiff.tensor...")
tic()
include("test_tensors.jl")
println("done (took $(toq()) seconds).")

print("Testing ForwardDiff.derivative...")
tic()
include("test_derivatives.jl")
println("done (took $(toq()) seconds).")

print("Testing ForwardDiff.jacobian...")
tic()
include("test_jacobians.jl")
println("done (took $(toq()) seconds).")
