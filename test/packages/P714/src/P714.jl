module P714

using ForwardDiff

dispatch(::Val{0}, x) = x
dispatch(::Val{1}, x) = ForwardDiff.derivative(z -> x + z^2, x)

# prevents precompilation of `dispatch`
indirection = dispatch

compute_derivative(α::Int, y) = ForwardDiff.derivative(x -> indirection(Val{α}(), x), y)

ForwardDiff.derivative(x -> x^2, 1)
compute_derivative(0, 0)

end
