f(x) = sum(sin, x) * x[1]^2 + exp(x[end])
g(x) = x .* sin(x[1]) .+ reverse(x)
inputs(n) = collect(range(0.1, 1.0; length = n))
results(x) = (ForwardDiff.gradient(f, x), ForwardDiff.jacobian(g, x), ForwardDiff.hessian(f, x))
