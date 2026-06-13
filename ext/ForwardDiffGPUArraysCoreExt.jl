module ForwardDiffGPUArraysCoreExt

using ForwardDiff: ForwardDiff, Dual, Partials
using GPUArraysCore: AbstractGPUArray

# ForwardDiff's default `seed!` methods (src/apiutils.jl) write each dual with a
# scalar `setindex!` loop over `structural_eachindex`. On GPU arrays that
# triggers a scalar-indexing error. GPU arrays are always dense, one-based, and
# carry isbits element types, so the structural-index / unset-element handling of
# the generic methods is unnecessary here and broadcast restores the
# pre-1.0 GPU-compatible behavior.

function ForwardDiff.seed!(
        duals::AbstractGPUArray{Dual{T,V,N}}, x,
        seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    duals .= Dual{T,V,N}.(x, Ref(seed))
    return duals
end

function ForwardDiff.seed!(
        duals::AbstractGPUArray{Dual{T,V,N}}, x,
        seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    dual_inds = 1:N
    duals[dual_inds] .= Dual{T,V,N}.(view(x, dual_inds), seeds)
    return duals
end

function ForwardDiff.seed!(
        duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
        seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    dual_inds = index:length(duals)
    duals[dual_inds] .= Dual{T,V,N}.(view(x, dual_inds), Ref(seed))
    return duals
end

function ForwardDiff.seed!(
        duals::AbstractGPUArray{Dual{T,V,N}}, x, index,
        seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    dual_inds = (1 + offset):(offset + chunksize)
    duals[dual_inds] .= Dual{T,V,N}.(view(x, dual_inds), seeds[1:chunksize])
    return duals
end

end
