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
    seed_zero_partials!(ydual, y)
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
structural_eachindex(x::AbstractArray) = structural_eachindex(x, x)
function structural_eachindex(x::AbstractArray, y::AbstractArray)
    require_one_based_indexing(x, y)
    eachindex(x, y)
end
function structural_eachindex(x::UpperTriangular, y::AbstractArray)
    require_one_based_indexing(x, y)
    if size(x) != size(y)
        throw(DimensionMismatch())
    end
    n = size(x, 1)
    return (CartesianIndex(i, j) for j in 1:n for i in 1:j)
end
function structural_eachindex(x::LowerTriangular, y::AbstractArray)
    require_one_based_indexing(x, y)
    if size(x) != size(y)
        throw(DimensionMismatch())
    end
    n = size(x, 1)
    return (CartesianIndex(i, j) for j in 1:n for i in j:n)
end
function structural_eachindex(x::Diagonal, y::AbstractArray)
    require_one_based_indexing(x, y)
    if size(x) != size(y)
        throw(DimensionMismatch())
    end
    return diagind(x)
end

@inline function dense_seedable(duals, x, ::Type{V}) where {V}
    return duals isa DenseArray && isbitstype(V) && !Base.has_offset_axes(duals, x)
end

struct SeededDual{D,S}
    seeds::S
    offset::Int
end

@inline (f::SeededDual{D})(x) where {D} = D(x, f.seeds)
@inline (f::SeededDual{D})(x, i) where {D} = D(x, f.seeds[i - f.offset])

# Copies the values of `x` into `duals` with zero partials. Used both to remove seeds `duals` is
# currently carrying and to initialize a freshly allocated work buffer, whose elements must all be
# written before the target function reads them.
function seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x) where {T,V,N}
    seed = zero(Partials{N,V})
    if dense_seedable(duals, x, V) && axes(duals) == axes(x)
        duals .= Dual{T,V,N}.(x, Ref(seed))
        return duals
    end
    return _seed_zero_partials!(duals, x, structural_eachindex(duals, x))
end

# Zeroes the partials of `count` elements starting at structural position `index`. Chunk mode only
# needs to clear the chunk it just seeded, so writing through to the end of the array would be O(n)
# redundant work per chunk, i.e. O(n^2/N) per sweep. `count` mirrors the `chunksize` argument of
# `seed!(duals, x, index, seeds, chunksize)`.
function seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, index,
                             count = N) where {T,V,N}
    if dense_seedable(duals, x, V)
        length(duals) == length(x) || throw(DimensionMismatch())
        last_index = min(index + count - 1, length(duals))
        dual_inds = index:last_index
        seed = zero(Partials{N,V})
        f = SeededDual{Dual{T,V,N},typeof(seed)}(seed, 0)
        map!(f, view(duals, dual_inds), view(x, dual_inds))
        return duals
    end
    idxs = Iterators.take(Iterators.drop(structural_eachindex(duals, x), index - 1), count)
    return _seed_zero_partials!(duals, x, idxs)
end

function _seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, idxs) where {T,V,N}
    seed = zero(Partials{N,V})
    if isbitstype(V)
        for idx in idxs
            duals[idx] = Dual{T,V,N}(x[idx], seed)
        end
    else
        for idx in idxs
            if isassigned(x, idx)
                duals[idx] = Dual{T,V,N}(x[idx], seed)
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    if dense_seedable(duals, x, V)
        length(duals) == length(x) || throw(DimensionMismatch())
        dual_inds = 1:min(N, length(duals))
        f = SeededDual{Dual{T,V,N},typeof(seeds)}(seeds, 0)
        map!(f, view(duals, dual_inds), view(x, dual_inds), dual_inds)
    elseif isbitstype(V)
        for (i, idx) in zip(1:N, structural_eachindex(duals, x))
            duals[idx] = Dual{T,V,N}(x[idx], seeds[i])
        end
    else
        for (i, idx) in zip(1:N, structural_eachindex(duals, x))
            if isassigned(x, idx)
                duals[idx] = Dual{T,V,N}(x[idx], seeds[i])
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    if dense_seedable(duals, x, V)
        length(duals) == length(x) || throw(DimensionMismatch())
        shift = index - 1
        dual_inds = (1 + shift):min(shift + chunksize, length(duals))
        f = SeededDual{Dual{T,V,N},typeof(seeds)}(seeds, shift)
        map!(f, view(duals, dual_inds), view(x, dual_inds), dual_inds)
        return duals
    end
    offset = index - 1
    idxs = Iterators.drop(structural_eachindex(duals, x), offset)
    if isbitstype(V)
        for (i, idx) in zip(1:chunksize, idxs)
            duals[idx] = Dual{T,V,N}(x[idx], seeds[i])
        end
    else
        for (i, idx) in zip(1:chunksize, idxs)
            if isassigned(x, idx)
                duals[idx] = Dual{T,V,N}(x[idx], seeds[i])
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end
