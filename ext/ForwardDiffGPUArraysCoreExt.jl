module ForwardDiffGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray
using ForwardDiff: ForwardDiff, Dual, Partials, npartials, partials

struct PartialsFn{T,D<:Dual}
    dual::D
end
PartialsFn{T}(dual::Dual) where {T} = PartialsFn{T,typeof(dual)}(dual)

(f::PartialsFn{T})(i) where {T} = partials(T, f.dual, i)

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x,
                           seed::Partials{N,V}) where {T,V,N}
    idxs = collect(ForwardDiff.structural_eachindex(duals, x))
    duals[idxs] .= Dual{T,V,N}.(view(x, idxs), Ref(seed))
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x,
                           seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    idxs = collect(Iterators.take(ForwardDiff.structural_eachindex(duals, x), N))
    duals[idxs] .= Dual{T,V,N}.(view(x, idxs), getindex.(Ref(seeds), 1:length(idxs)))
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
                           seed::Partials{N,V}) where {T,V,N}
    offset = index - 1
    idxs = collect(Iterators.drop(ForwardDiff.structural_eachindex(duals, x), offset))
    duals[idxs] .= Dual{T,V,N}.(view(x, idxs), Ref(seed))
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
                           seeds::NTuple{N,Partials{N,V}}, chunksize) where {T,V,N}
    offset = index - 1
    idxs = collect(
        Iterators.take(Iterators.drop(ForwardDiff.structural_eachindex(duals, x), offset), chunksize)
    )
    duals[idxs] .= Dual{T,V,N}.(view(x, idxs), getindex.(Ref(seeds), 1:length(idxs)))
    return duals
end

# gradient
function ForwardDiff.extract_gradient!(::Type{T}, result::AbstractGPUArray,
                                       dual::Dual) where {T}
    fn = PartialsFn{T}(dual)
    idxs = collect(Iterators.take(ForwardDiff.structural_eachindex(result), npartials(dual)))
    result[idxs] .= fn.(1:length(idxs))
    return result
end

function ForwardDiff.extract_gradient_chunk!(::Type{T}, result::AbstractGPUArray, dual,
                                             index, chunksize) where {T}
    fn = PartialsFn{T}(dual)
    offset = index - 1
    idxs = collect(
        Iterators.take(Iterators.drop(ForwardDiff.structural_eachindex(result), offset), chunksize)
    )
    result[idxs] .= fn.(1:length(idxs))
    return result
end

end
