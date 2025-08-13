module ForwardDiffGPUArraysCoreExt

using GPUArraysCore: AbstractGPUArray
using ForwardDiff: ForwardDiff, Dual, Partials

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x,
                           seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    idxs = ForwardDiff.structural_eachindex(duals, x)
    duals[idxs] .= Dual{T,V,N}.(x[idxs], seed)
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x,
                           seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    idxs = ForwardDiff.structural_eachindex(duals, x)
    duals[idxs] .= Dual{T,V,N}.(x[idxs], seeds[1:N])
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
                           seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    offset = index - 1
    idxs = Iterators.drop(ForwardDiff.structural_eachindex(duals, x), offset)
    duals[idxs] .= Dual{T,V,N}.(x[idxs], seed)
    return duals
end

function ForwardDiff.seed!(duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
                           seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    idxs = Iterators.drop(ForwardDiff.structural_eachindex(duals, x), offset)
    duals[idxs] .= Dual{T,V,N}.(x[idxs], seeds[1:chunksize])
    return duals
end

end
