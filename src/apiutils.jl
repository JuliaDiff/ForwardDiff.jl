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
    check_matching_size(x, y)
    n = size(x, 1)
    return (CartesianIndex(i, j) for j in 1:n for i in 1:j)
end
function structural_eachindex(x::LowerTriangular, y::AbstractArray)
    require_one_based_indexing(x, y)
    check_matching_size(x, y)
    n = size(x, 1)
    return (CartesianIndex(i, j) for j in 1:n for i in j:n)
end
function structural_eachindex(x::Diagonal, y::AbstractArray)
    require_one_based_indexing(x, y)
    check_matching_size(x, y)
    return diagind(x)
end

# The two arrays are indexed by the same indices, so they have to have the same size. This is the
# error a `gradient!` into a container that is not shaped like `x` aborts with, so it names the sizes.
function check_matching_size(x::AbstractArray, y::AbstractArray)
    size(x) == size(y) || throw(DimensionMismatch(
        lazy"expected an array of size $(size(x)), got an array of size $(size(y))"))
    return nothing
end

# The columns of the Jacobian `out` that receive derivatives, in seeding order. Column `j` holds the
# derivatives with respect to `x[j]`, so a seeded entry writes to the column at its position in the
# linear order of `x`. Every entry of an array is seeded unless one of the methods below applies.
function structural_columns(out::AbstractMatrix, x::AbstractArray)
    require_one_based_indexing(out, x)
    check_matching_columns(out, x)
    return axes(out, 2)
end
# The seeded columns are runs of increasing length, so they are no range. Deriving their order from
# `structural_eachindex` rather than recomputing it keeps a single source of truth: the two have to
# agree entry by entry, or the derivatives land in the wrong columns.
function structural_columns(out::AbstractMatrix, x::Union{LowerTriangular,UpperTriangular})
    require_one_based_indexing(out, x)
    check_matching_columns(out, x)
    cols = axes(out, 2)
    lin = LinearIndices(x)
    return (cols[lin[idx]] for idx in structural_eachindex(x))
end
# `diagind` is already a range of linear positions, so it can select the columns directly.
function structural_columns(out::AbstractMatrix, x::Diagonal)
    require_one_based_indexing(out, x)
    check_matching_columns(out, x)
    return axes(out, 2)[diagind(x)]
end

# A column of the Jacobian belongs to an entry of `x`, so there have to be as many as `x` has entries.
function check_matching_columns(out::AbstractMatrix, x::AbstractArray)
    size(out, 2) == length(x) || throw(DimensionMismatch(
        lazy"expected a matrix with $(length(x)) columns, got a matrix with $(size(out, 2)) columns"))
    return nothing
end

# `structural_columns` is lazy, since walking it once from the front is all `jacobian!` ever needs.
# The Hessian sweep reads the positions of two blocks at once and re-reads each row block once per
# column block, so it indexes into them instead, and materializes them first. The ranges of
# unstructured inputs and of `Diagonal` are already indexable and pass through unchanged; only the
# triangular wrappers pay, one `structural_length(x)`-element vector against an n²-entry result.
_indexable(idxs::AbstractArray) = idxs
_indexable(idxs) = collect(idxs)

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
                Base._unsetindex!(duals, idx)
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

# Seed a chunk in either layer of nested duals. A `nothing` seed clears that layer. This is not
# `seed_zero_partials!` with both layers cleared: that builds the inner value at the exact type of
# the array's `V`, which the nested buffer's `Dual{T,V,N}` is not, so it would be a `MethodError`.
function seed_hessian_chunk!(duals::AbstractArray{Dual{T,Dual{T,V,N},N}}, x, index,
                             iseeds::Union{Nothing,NTuple{N,Partials{N,V}}},
                             oseeds::Union{Nothing,NTuple{N,Partials{N,Dual{T,V,N}}}},
                             chunksize = N) where {T,V,N}
    # `iseeds === nothing` is a compile-time constant here, so building each layer's accessor in its
    # own branch means the zero partials it does not need are never constructed. That is free for an
    # isbits `V` but not, say, for `BigFloat`.
    ipartials = iseeds === nothing ? Returns(zero(Partials{N,V})) : Base.Fix1(getindex, iseeds)
    opartials = oseeds === nothing ? Returns(zero(Partials{N,Dual{T,V,N}})) : Base.Fix1(getindex, oseeds)
    idxs = Iterators.take(Iterators.drop(structural_eachindex(duals, x), index - 1), chunksize)
    return _seed!(duals, x, idxs) do value, i
        Dual{T,Dual{T,V,N},N}(Dual{T,V,N}(value, ipartials(i)), opartials(i))
    end
end
