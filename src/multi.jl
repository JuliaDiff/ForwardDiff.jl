"""
    ForwardDiff.multigrad(f, xs...)

Return a tuple containing the components of the gradient, 
`∂f/∂x` for each argument `x ∈ xs`. Requires that `f(xs...) isa Real`.

These are arrays for array arguments, using [`gradient`](@ref),
and numbers for scalar arguments, using [`derivative`](@ref).

Any other type of argument is assumed not to be differentiable, 
indicated by `nothing`. Any keyword arguments are likewise not 
differentiated, but are passed to the function: `f(xs...; kw...)`.
"""
function multigrad(f, xs...; kw...)
    ntuple(length(xs)) do i
        g = y -> f(ntuple(j -> j==i ? y : xs[j], length(xs))...; kw...)
        x = xs[i]
        if x isa AbstractArray
            ForwardDiff.gradient(g, xs[i])
        elseif x isa Number
            ForwardDiff.derivative(g, xs[i])
        else
            nothing
        end
    end
end

multigrad(f, xs::Real...; kw...) = multigrad(f, promote(xs...)...; kw...)

function multigrad(f, xs::Vararg{T,N}; kw...) where {T<:Real,N}
    args = ntuple(N) do i
        Dual(xs[i], ntuple(j -> T(j==i), N))
    end
    y = f(args...; kw...)
    y isa Number || throw(DimensionMismatch(
        "multigrad(f, xs...) expects that f return a real number. Perhaps you meant multijacobi(f, x)?"))
    Tuple(partials(y))
end

"""
    ForwardDiff.multijacobi(f, xs...)

Return a tuple containing the components of the Jacobian, 
`∂f/∂x` for each argument `x ∈ xs`. Requires that `f(xs...) isa AbstractArray`.

For array arguments `x` these are matrices, using [`jacobian`](@ref),
and for scalar arguments they are vectors, using [`vec(derivative(...))`](@ref).

Any other type of argument is assumed not to be differentiable, 
indicated by `nothing`. Any keyword arguments are likewise not 
differentiated, but are passed to the function: `f(xs...; kw...)`.
"""
function multijacobi(f, xs...; kw...)
    ntuple(length(xs)) do i
        g = y -> f(ntuple(j -> j==i ? y : xs[j], length(xs))...; kw...)
        x = xs[i]
        if x isa AbstractArray
            ForwardDiff.jacobian(g, xs[i])
        elseif x isa Number
            vec(ForwardDiff.derivative(g, xs[i]))
        else
            nothing
        end
    end
end
