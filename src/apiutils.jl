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
    seed!(eltype(cfg), xdual, x, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval!(f!::F, cfg::JacobianConfig{T,V,N}, y, x) where {F,T,V,N}
    ydual, xdual = cfg.duals
    seed!(eltype(cfg), xdual, x, cfg.seeds)
    seed_zero_partials!(Dual{T,eltype(y),N}, ydual, y)
    f!(ydual, xdual)
    return ydual
end

##################################
# seed construction/manipulation #
##################################

@generated function construct_seeds(::Type{Partials{N,V}}) where {N,V}
    return Expr(:tuple, [:(single_seed(Partials{N,V}, Val{$i}())) for i in 1:N]...)
end

# Seeds `x` with tag `T` and partials `p`, converted to the type of `x`. Layers of `x` with greater
# tags are kept outside, so nested `Dual`s stay sorted even if `T` is not greater than all tags in `x`.
@inline seed_dual(::Type{T}, x, p::Partials{N}) where {T,N} = Dual{T}(x, convert(Partials{N,typeof(x)}, p))
@inline function seed_dual(::Type{T}, x::Dual{S}, p::Partials{N}) where {T,S,N}
    p = convert(Partials{N,typeof(x)}, p)
    T ≺ S || return Dual{T}(x, p)
    # the seeds are constants, so their partials w.r.t. `S` are zero
    q = map_partials(y -> value(S, y), valtype(S, eltype(p)), p)
    return Dual{S}(seed_dual(T, value(S, x), q), map(y -> seed_dual(T, y, zero(q)), partials(S, x).values))
end

# Type of `seed_dual(T, x, p)` for `x::V` and `p::Partials{N,V}`. If `V` is abstract, each element is
# seeded with its own type, so only `Real` is a bound.
seed_type(::Type{Dual{T,V,N}}) where {T,V,N} = isconcretetype(V) ? Dual{T,V,N} : Real
function seed_type(::Type{Dual{T,Dual{S,W,M},N}}) where {T,S,W,M,N}
    return T ≺ S ? Dual{S,seed_type(Dual{T,W,N}),M} : Dual{T,Dual{S,W,M},N}
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

# Copies the values of `x` into `duals` with zero partials. Used both to remove seeds `duals` is
# currently carrying and to initialize a freshly allocated work buffer, whose elements must all be
# written before the target function reads them.
seed_zero_partials!(::Type{D}, duals::AbstractArray, x) where {D<:Dual} =
    _seed_zero_partials!(D, duals, x, structural_eachindex(duals, x))

# Zeroes the partials of `count` elements starting at structural position `index`. Chunk mode only
# needs to clear the chunk it just seeded, so writing through to the end of the array would be O(n)
# redundant work per chunk, i.e. O(n^2/N) per sweep. `count` mirrors the `chunksize` argument of
# `seed!(D, duals, x, index, seeds, chunksize)`.
function seed_zero_partials!(::Type{Dual{T,V,N}}, duals::AbstractArray, x, index,
                             count = N) where {T,V,N}
    idxs = Iterators.take(Iterators.drop(structural_eachindex(duals, x), index - 1), count)
    return _seed_zero_partials!(Dual{T,V,N}, duals, x, idxs)
end

function _seed_zero_partials!(::Type{Dual{T,V,N}}, duals::AbstractArray, x, idxs) where {T,V,N}
    seed = zero(Partials{N,V})
    if isbitstype(V)
        for idx in idxs
            duals[idx] = seed_dual(T, x[idx], seed)
        end
    else
        for idx in idxs
            if isassigned(x, idx)
                duals[idx] = seed_dual(T, x[idx], seed)
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end

function seed!(::Type{Dual{T,V,N}}, duals::AbstractArray, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    if isbitstype(V)
        for (i, idx) in zip(1:N, structural_eachindex(duals, x))
            duals[idx] = seed_dual(T, x[idx], seeds[i])
        end
    else
        for (i, idx) in zip(1:N, structural_eachindex(duals, x))
            if isassigned(x, idx)
                duals[idx] = seed_dual(T, x[idx], seeds[i])
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end

function seed!(::Type{Dual{T,V,N}}, duals::AbstractArray, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    idxs = Iterators.drop(structural_eachindex(duals, x), offset)
    if isbitstype(V)
        for (i, idx) in zip(1:chunksize, idxs)
            duals[idx] = seed_dual(T, x[idx], seeds[i])
        end
    else
        for (i, idx) in zip(1:chunksize, idxs)
            if isassigned(x, idx)
                duals[idx] = seed_dual(T, x[idx], seeds[i])
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end
