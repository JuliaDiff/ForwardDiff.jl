module ForwardDiffGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray
using ForwardDiff: ForwardDiff, Dual, Partials, npartials, partials

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x,
                           seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    idxs = collect(ForwardDiff.structural_eachindex(duals, x))
    duals[idxs] .= Dual{T,V,N}.(x[idxs], Ref(seed))
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x,
                           seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    idxs = collect(ForwardDiff.structural_eachindex(duals, x))[1:N]
    duals[idxs] .= Dual{T,V,N}.(x[idxs], seeds[1:N])
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
                           seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    offset = index - 1
    idxs = collect(Iterators.drop(ForwardDiff.structural_eachindex(duals, x), offset))
    duals[idxs] .= Dual{T,V,N}.(x[idxs], Ref(seed))
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
                           seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    idxs = collect(
        Iterators.drop(ForwardDiff.structural_eachindex(duals, x), offset)
    )[1:chunksize]
    duals[idxs] .= Dual{T,V,N}.(x[idxs], seeds[1:chunksize])
    return duals
end

# gradient
function ForwardDiff.extract_gradient!(::Type{T}, result::AbstractGPUArray,
                                       dual::Dual) where {T}
    # this closure is needed for gpu compilation
    partial_fn(dual, i) = partials(T, dual, i)

    idxs = ForwardDiff.structural_eachindex(result)
    result[idxs] .= partial_fn.(Ref(dual), 1:npartials(dual))
    return result
end

function ForwardDiff.extract_gradient_chunk!(::Type{T}, result::AbstractGPUArray, dual,
                                             index, chunksize) where {T}
    # this closure is needed for gpu compilation
    partial_fn(dual, i) = partials(T, dual, i)

    offset = index - 1
    idxs = collect(
        Iterators.drop(ForwardDiff.structural_eachindex(result), offset)
    )[1:chunksize]
    result[idxs] .= partial_fn.(Ref(dual), 1:chunksize)
    return result
end

end
