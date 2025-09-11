# overload for array types that
supports_fast_scalar_indexing(::Array) = true

function supports_fast_scalar_indexing(x::AbstractArray)
    return parent(x) !== x && supports_fast_scalar_indexing(parent(x))
end

# Helper function for broadcasting
struct PartialsFn{T,D<:Dual}
    dual::D
end
PartialsFn{T}(dual::Dual) where {T} = PartialsFn{T,typeof(dual)}(dual)

(f::PartialsFn{T})(i) where {T} = partials(T, f.dual, i)
