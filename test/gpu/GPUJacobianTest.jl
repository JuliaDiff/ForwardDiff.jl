using ForwardDiff, CUDA, Test

f(x) = x .^ 2 ./ 2

x = [1.0, 2.0, 3.0]
x_jl = CuArray(x)

jac = ForwardDiff.jacobian(f, x)
jac_jl = ForwardDiff.jacobian(f, x_jl)

@test jac_jl isa CuArray
@test Array(jac_jl) ≈ jac