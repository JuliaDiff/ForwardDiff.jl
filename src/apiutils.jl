####################
# value extraction #
####################

@inline extract_value!(::Type{T}, out::DiffResult, ydual) where {T} =
    DiffResults.value!(d -> value(T,d), out, ydual)
@inline extract_value!(::Type{T}, out, ydual) where {T} = out # ???

@inline function extract_value!(::Type{T}, out, y, ydual) where {T}
    map!(d -> value(T,d), y, ydual)
    copy_value!(out, y)
end

@inline copy_value!(out::DiffResult, y) = DiffResults.value!(out, y)
@inline copy_value!(out, y) = out

###################################
# vector mode function evaluation #
###################################

function vector_mode_dual_eval!(f::F, cfg::Union{JacobianConfig,GradientConfig}, x) where {F}
    xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval!(f!::F, cfg::JacobianConfig, y, x) where {F}
    ydual, xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
    return ydual
end

##################################
# seed construction/manipulation #
##################################

@generated function construct_seeds(::Type{Partials{N,V}}) where {N,V}
    return Expr(:tuple, [:(single_seed(Partials{N,V}, Val{$i}())) for i in 1:N]...)
end

# Only seed indices that are structurally non-zero
_structural_nonzero_indices(x::AbstractArray) = eachindex(x)
function _structural_nonzero_indices(x::UpperTriangular)
    n = size(x, 1)
    return (CartesianIndex(i, j) for j in 1:n for i in 1:j)
end
function _structural_nonzero_indices(x::LowerTriangular)
    n = size(x, 1)
    return (CartesianIndex(i, j) for j in 1:n for i in j:n)
end
_structural_nonzero_indices(x::Diagonal) = diagind(x)

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    if eachindex(duals) != eachindex(x)
        throw(ArgumentError("indices of input array and array of duals are not identical"))
    end
    for idx in _structural_nonzero_indices(duals)
        duals[idx] = Dual{T,V,N}(x[idx], seed)
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    if eachindex(duals) != eachindex(x)
        throw(ArgumentError("indices of input array and array of duals are not identical"))
    end
    for (i, idx) in enumerate(_structural_nonzero_indices(duals))
        duals[idx] = Dual{T,V,N}(x[idx], seeds[i])
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    if eachindex(duals) != eachindex(x)
        throw(ArgumentError("indices of input array and array of duals are not identical"))
    end
    offset = index - 1
    idxs = Iterators.drop(_structural_nonzero_indices(duals), offset)
    for idx in idxs
        duals[idx] = Dual{T,V,N}(x[idx], seed)
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    if eachindex(duals) != eachindex(x)
        throw(ArgumentError("indices of input array and array of duals are not identical"))
    end
    offset = index - 1
    idxs = Iterators.drop(_structural_nonzero_indices(duals), offset)
    for (i, idx) in enumerate(idxs)
        i > chunksize && break
        duals[idx] = Dual{T,V,N}(x[idx], seeds[i])
    end
    return duals
end
