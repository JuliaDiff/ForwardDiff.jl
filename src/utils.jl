# overload for array types that
@inline supports_fast_scalar_indexing(::Array) = true

@inline function supports_fast_scalar_indexing(x::AbstractArray)
    parent(x) === x && return false
    return supports_fast_scalar_indexing(parent(x))
end

# Helper function for broadcasting
struct PartialsFn{T,D<:Dual}
    dual::D
end
PartialsFn{T}(dual::Dual) where {T} = PartialsFn{T,typeof(dual)}(dual)

(f::PartialsFn{T})(i) where {T} = partials(T, f.dual, i)
