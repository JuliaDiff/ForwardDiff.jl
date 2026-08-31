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

function check_structural_size(duals, x)
    if size(duals) != size(x)
        throw(DimensionMismatch(lazy"the config was built for an array of size $(size(duals)) and cannot be used with an array of size $(size(x))"))
    end
    return nothing
end

# The positions of `structural_eachindex`, in the same order, as linear indices of `x`. The two
# argument form is only ever given a config's work buffer and the input it is used with.
structural_linearindices(x::AbstractArray) = structural_linearindices(x, x)
function structural_linearindices(duals::AbstractArray, x::AbstractArray)
    require_one_based_indexing(duals, x)
    check_structural_size(duals, x)
    return Base.OneTo(length(duals))
end
function structural_linearindices(duals::UpperTriangular, x::AbstractArray)
    require_one_based_indexing(duals, x)
    check_structural_size(duals, x)
    n = size(duals, 1)
    return [i + n * (j - 1) for j in 1:n for i in 1:j]
end
function structural_linearindices(duals::LowerTriangular, x::AbstractArray)
    require_one_based_indexing(duals, x)
    check_structural_size(duals, x)
    n = size(duals, 1)
    return [i + n * (j - 1) for j in 1:n for i in j:n]
end
function structural_linearindices(duals::Diagonal, x::AbstractArray)
    require_one_based_indexing(duals, x)
    check_structural_size(duals, x)
    n = size(duals, 1)
    return range(1; step = n + 1, length = n)
end

# The `count` positions starting at structural position `index`.
structural_chunk(indices, index, count) = view(indices, index:(index + count - 1))

# Copies the values of `x` into `duals` with zero partials. Used both to remove seeds `duals` is
# currently carrying and to initialize a freshly allocated work buffer, whose elements must all be
# written before the target function reads them.
seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x) where {T,V,N} =
    _seed_zero_partials!(duals, x, structural_eachindex(duals, x))

# Zeroes the partials of `count` elements starting at structural position `index`. Chunk mode only
# needs to clear the chunk it just seeded, so writing through to the end of the array would be O(n)
# redundant work per chunk, i.e. O(n^2/N) per sweep. `count` mirrors the `chunksize` argument of
# `seed!(duals, x, index, seeds, chunksize)`.
function seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, index,
                             count = N) where {T,V,N}
    idxs = Iterators.take(Iterators.drop(structural_eachindex(duals, x), index - 1), count)
    return _seed_zero_partials!(duals, x, idxs)
end

function _seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, idxs) where {T,V,N}
    seed = zero(Partials{N,V})
    return _seed!(duals, x, idxs) do value, _
        Dual{T,V,N}(value, seed)
    end
end

# `Base._unsetindex!` is implemented for `Array` alone: for a linear index its `AbstractArray`
# fallback recurses forever, and it has no `CartesianIndex` method at all.
_unsetindex!(duals::Array, idx) = Base._unsetindex!(duals, idx)
_unsetindex!(duals::AbstractArray, idx) = throw(ArgumentError(LazyString(
    "cannot differentiate at an input with an unassigned entry at index ", idx,
    ": that would leave an entry of the ", nameof(typeof(duals)),
    " work buffer unassigned, which is only possible for an Array")))

# Write a sequence of duals while preserving unassigned entries in arrays whose element type is not
# stored inline. `make_dual` receives the primal value and its one-based position in `idxs`.
@inline function _seed!(make_dual::F, duals::AbstractArray{Dual{T,V,N}}, x, idxs) where {F,T,V,N}
    if isbitstype(V)
        for (i, idx) in enumerate(idxs)
            duals[idx] = make_dual(x[idx], i)
        end
    else
        for (i, idx) in enumerate(idxs)
            if isassigned(x, idx)
                duals[idx] = make_dual(x[idx], i)
            else
                _unsetindex!(duals, idx)
            end
        end
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    idxs = Iterators.take(structural_eachindex(duals, x), N)
    return _seed!(duals, x, idxs) do value, i
        Dual{T,V,N}(value, seeds[i])
    end
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    idxs = Iterators.take(Iterators.drop(structural_eachindex(duals, x), offset), chunksize)
    return _seed!(duals, x, idxs) do value, i
        Dual{T,V,N}(value, seeds[i])
    end
end

# Seed a chunk in either layer of nested duals. A `nothing` seed clears that layer;
# `seed_zero_partials!` cannot, as it would pass the primal where a nested `Dual` is wanted.
function seed_hessian_chunk!(duals::AbstractArray{Dual{T,Dual{T,V,N},N}}, x, indices, index,
                             iseeds::Union{Nothing,NTuple{N,Partials{N,V}}},
                             oseeds::Union{Nothing,NTuple{N,Partials{N,Dual{T,V,N}}}},
                             chunksize = N) where {T,V,N}
    izero = iseeds === nothing ? zero(Partials{N,V}) : nothing
    ozero = oseeds === nothing ? zero(Partials{N,Dual{T,V,N}}) : nothing
    idxs = structural_chunk(indices, index, chunksize)
    return _seed!(duals, x, idxs) do value, i
        inner = Dual{T,V,N}(value, iseeds === nothing ? izero : iseeds[i])
        Dual{T,Dual{T,V,N},N}(inner, oseeds === nothing ? ozero : oseeds[i])
    end
end
