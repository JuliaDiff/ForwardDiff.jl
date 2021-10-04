#####################################
# AbstractDifferentiation interface #
#####################################

struct ForwardDiffBackend <: AD.AbstractBackend end

AD.@primitive function pushforward_function(ba::ForwardDiffBackend, f, xs...)
    return function pushforward(vs)
        return derivative(h -> f(step_toward.(xs, vs, h)...), 0)
    end
end
function AD.pushforward_function(::ForwardDiffBackend, f, x)
    return function pushforward(v)
        if v isa Tuple
            @assert length(v) == 1
            return (derivative(h -> f(step_toward(x, v[1], h)), 0),)
        else
            return (derivative(h -> f(step_toward(x, v, h)), 0),)
        end
    end
end

AD.primal_value(::ForwardDiffBackend, ::Any, f, xs) = value.(f(xs...))

# these implementations are more efficient than the fallbacks in AbstractDifferentiation

AD.gradient(::ForwardDiffBackend, f, x::AbstractArray) = (gradient(f, x),)

AD.jacobian(::ForwardDiffBackend, f, x::AbstractArray) = (jacobian(at_least_0dim ∘ f, x),)
AD.jacobian(::ForwardDiffBackend, f, x::Number) = (derivative(f, x),)

AD.hessian(::ForwardDiffBackend, f, x::AbstractArray) = (hessian(f, x),)

function AD.value_and_gradient(::ForwardDiffBackend, f, x::AbstractArray)
    result = gradient!(DiffResults.GradientResult(x), f, x)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_jacobian(::ForwardDiffBackend, f, x::Number)
    result = derivative!(DiffResults.DiffResult(x, x), f, x)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end
function AD.value_and_jacobian(::ForwardDiffBackend, f, xs::Number...)
    xs_vec = SVector(xs...)
    result = gradient!(DiffResults.GradientResult(xs_vec), xs -> f(xs...), xs_vec)
    return DiffResults.value(result), Tuple(DiffResults.derivative(result))
end

function AD.value_and_hessian(::ForwardDiffBackend, f, x)
    result = hessian!(DiffResults.HessianResult(x), f, x)
    return DiffResults.value(result), (DiffResults.hessian(result),)
end

function AD.value_gradient_and_hessian(::ForwardDiffBackend, f, x)
    result = hessian!(DiffResults.HessianResult(x), f, x)
    return (
        DiffResults.value(result),
        (DiffResults.gradient(result),),
        (DiffResults.hessian(result),),
    )
end

@inline at_least_0dim(x::Number) = fill(x)
@inline at_least_0dim(x) = x

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v
