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

@inline step_toward(x::Number, v::Number, h) = x + h * v
# support arrays and tuples
@noinline step_toward(x, v, h) = x .+ h .* v
