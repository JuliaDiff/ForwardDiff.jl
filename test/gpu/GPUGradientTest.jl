using ForwardDiff, CUDA, Test

fn(x) = sum(x .^ 2 ./ 2)

x = [1.0, 2.0, 3.0]
x_jl = CuArray(x)

grad = ForwardDiff.gradient(fn, x)
grad_jl = ForwardDiff.gradient(fn, x_jl)

@test grad_jl isa CuArray
@test Array(grad_jl) ≈ grad

cfg = ForwardDiff.GradientConfig(
    fn, x_jl, ForwardDiff.Chunk{2}(), ForwardDiff.Tag(fn, eltype(x))
)
grad_jl = ForwardDiff.gradient(fn, x_jl, cfg)

@test grad_jl isa CuArray
@test Array(grad_jl) ≈ grad