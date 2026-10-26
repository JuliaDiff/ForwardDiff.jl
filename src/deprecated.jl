# Accessors without a tag extract the outermost layer of a nested `Dual`, which depends on
# the order of the tags rather than on the derivative the caller is interested in.

function depwarn_untagged(f::Symbol, replacement::String)
    Base.depwarn("`ForwardDiff.$f` without a tag is deprecated, use `$replacement` with the tag `T` of the derivative instead.", f)
end

@inline outer_partials(x, i) = zero(x)
@inline outer_partials(d::Dual{T}, i) where {T} = partials(T, d, i)
@inline outer_partials(x, i, j, k...) = outer_partials(outer_partials(x, i), j, k...)

function value(x)
    depwarn_untagged(:value, "ForwardDiff.value(T, x)")
    return x
end
function value(d::Dual{T}) where {T}
    depwarn_untagged(:value, "ForwardDiff.value(T, d)")
    return value(T, d)
end

function partials(x)
    depwarn_untagged(:partials, "ForwardDiff.partials(T, x)")
    return Partials{0,typeof(x)}(tuple())
end
function partials(d::Dual{T}) where {T}
    depwarn_untagged(:partials, "ForwardDiff.partials(T, d)")
    return partials(T, d)
end
function partials(x, i, j...)
    depwarn_untagged(:partials, "ForwardDiff.partials(T, x, i)")
    return outer_partials(x, i, j...)
end
function partials(::Type{T}, x, i, j, k...) where {T}
    Base.depwarn("`ForwardDiff.partials(T, x, i, j...)` is deprecated, use `ForwardDiff.partials(S, ForwardDiff.partials(T, x, i), j)` with the tag `S` of the inner derivative instead.", :partials)
    return outer_partials(partials(T, x, i), j, k...)
end

function npartials(::Dual{T,V,N}) where {T,V,N}
    depwarn_untagged(:npartials, "ForwardDiff.npartials(T, typeof(d))")
    return N
end
function npartials(::Type{Dual{T,V,N}}) where {T,V,N}
    depwarn_untagged(:npartials, "ForwardDiff.npartials(T, D)")
    return N
end
