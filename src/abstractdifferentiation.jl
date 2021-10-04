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

function AD.jacobian(ba::ForwardDiffBackend, f, x::AbstractArray)
    return AD.value_and_jacobian(ba, f, x)[2]
end
AD.jacobian(::ForwardDiffBackend, f, x::Number) = (derivative(f, x),)

AD.hessian(::ForwardDiffBackend, f, x::AbstractArray) = (hessian(f, x),)

function AD.value_and_gradient(::ForwardDiffBackend, f, x::AbstractArray)
    result = gradient!(DiffResults.GradientResult(x), f, x)
    return DiffResults.value(result), (DiffResults.derivative(result),)
end

function AD.value_and_jacobian(::ForwardDiffBackend, f, xs::AbstractArray)
    y = f(xs)
    if y isa Number
        return y, (adjoint(ForwardDiff.gradient(f, xs)),)
    else
        return y, (ForwardDiff.jacobian(f, xs),)
    end
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

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v
