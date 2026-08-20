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
    seed!(xdual, x, cfg.indices, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval!(f!::F, cfg::JacobianConfig, y, x) where {F}
    ydual, xdual = cfg.duals
    yindices, xindices = cfg.indices
    seed!(xdual, x, xindices, cfg.seeds)
    seed_zero_partials!(ydual, y, yindices)
    f!(ydual, xdual)
    return ydual
end

##################################
# seed construction/manipulation #
##################################

@generated function construct_seeds(::Type{Partials{N,V}}) where {N,V}
    return Expr(:tuple, [:(single_seed(Partials{N,V}, Val{$i}())) for i in 1:N]...)
end

########################
# structural positions #
########################

# Names which `structural_indices` method an array picks, so that two arrays can be checked for
# having the same structural positions without comparing the positions themselves. Add a method here
# whenever one is added below.
structural_kind(::AbstractArray) = nothing
structural_kind(::LowerTriangular) = LowerTriangular
structural_kind(::UpperTriangular) = UpperTriangular
structural_kind(::Diagonal) = Diagonal

# The linear indices of the entries that are seeded, in seeding order. Configs hold one of these per
# work buffer, so a sweep can index straight to a chunk instead of walking a lazy iterator from the
# front.
#
# The positions are linear indices of the array itself, so one position indexes the input, the work
# buffer and the result alike, and is a Jacobian column number as it stands -- the convention the
# outputs already follow.
#
# Only the triangles are materialized; the other two are ranges already.
function structural_indices(x::AbstractArray)
    require_one_based_indexing(x)
    return Base.OneTo(length(x))
end
function structural_indices(x::Diagonal)
    require_one_based_indexing(x)
    return diagind(x)
end
function structural_indices(x::UpperTriangular)
    require_one_based_indexing(x)
    n = size(x, 1)
    return [i + (j - 1) * n for j in 1:n for i in 1:j]
end
function structural_indices(x::LowerTriangular)
    require_one_based_indexing(x)
    n = size(x, 1)
    return [i + (j - 1) * n for j in 1:n for i in j:n]
end

# The positions of the `count` entries starting at structural position `index`. Allocation-free: a
# view of a range is a range, a view of a `Vector` a `SubArray`. A window overrunning the end is a
# `BoundsError` rather than a silently truncated chunk, i.e. silently missing derivatives.
structural_chunk(indices, index, count) = view(indices, index:(index + count - 1))

# Checks that the structural positions of `x` are positions of `y` as well. A dense pair may differ
# in shape -- extracting a gradient into a result of the same length has always worked -- but the
# positions of a structured `x` come from its size, so there `y` has to share it.
function check_structural_indices(x::AbstractArray, y::AbstractArray)
    require_one_based_indexing(y)
    length(x) == length(y) || throw(DimensionMismatch(
        lazy"expected an array with $(length(x)) elements, got an array with $(length(y)) elements"))
    return nothing
end
function check_structural_indices(x::Union{LowerTriangular,UpperTriangular,Diagonal},
                                  y::AbstractArray)
    require_one_based_indexing(y)
    return check_matching_size(x, y)
end

# The two arrays are indexed by the same indices, so they have to have the same size. This is the
# error a `gradient!` into a container that is not shaped like `x` aborts with, so it names the sizes.
function check_matching_size(x::AbstractArray, y::AbstractArray)
    size(x) == size(y) || throw(DimensionMismatch(
        lazy"expected an array of size $(size(x)), got an array of size $(size(y))"))
    return nothing
end

# A config's positions were built for its work buffer, which `similar` gave the structure and the size
# of the array the config was constructed for, so they are the positions of `x` too exactly when the
# two are structurally interchangeable. Comparing the kinds is O(1) and a compile-time constant, so
# this can run on every call.
function checkstructure(duals::AbstractArray, x::AbstractArray)
    structural_kind(duals) === structural_kind(x) || throw(ArgumentError(LazyString(
        "the config was built for an array of type ", nameof(typeof(duals)),
        " and cannot be used with an array of type ", nameof(typeof(x)),
        ": the two are structured differently")))
    return check_structural_indices(duals, x)
end

checkstructure(cfg::AbstractConfig, x) = checkstructure(cfg.duals, x)

# The `f!(y, x)` configs hold a buffer for the output as well, and it is seeded too.
function checkstructure(cfg::AbstractConfig, y, x)
    ydual, xdual = cfg.duals
    checkstructure(ydual, y)
    return checkstructure(xdual, x)
end

###########
# seeding #
###########

# Mirrors an unassigned entry of `x` into the work buffer. Only an `Array` owns the storage its
# entries live in, and `Base._unsetindex!` is implemented for nothing else -- a structured wrapper is
# a view onto a parent, so it has no slot of its own to unset. Base's `AbstractArray` fallback
# recurses until the stack runs out, so say so instead.
_unsetindex!(duals::Array, idx) = Base._unsetindex!(duals, idx)
_unsetindex!(duals::AbstractArray, idx) = throw(ArgumentError(LazyString(
    "cannot differentiate at an input with an unassigned entry at index ", idx,
    ": that would leave an entry of the ", nameof(typeof(duals)),
    " work buffer unassigned, which is only possible for an Array")))

# The two seeding operations below differ only in the `Dual` they build for the `i`th position of
# their window, so they share the walk. The `isbitstype` branch keeps the common case a plain store:
# `isassigned` is a `try`/`catch` for most array types, and a bits element type is never unassigned.
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

# Copies the values of `x` into `duals` with zero partials. Used both to remove seeds `duals` is
# currently carrying and to initialize a freshly allocated work buffer, whose elements must all be
# written before the target function reads them.
seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, indices) where {T,V,N} =
    _seed_zero_partials!(duals, x, indices)

# Zeroes the partials of `count` elements starting at structural position `index`. Chunk mode only
# needs to clear the chunk it just seeded, so writing through to the end of the array would be O(n)
# redundant work per chunk, i.e. O(n^2/N) per sweep. `count` mirrors the `chunksize` argument of
# `seed!(duals, x, indices, index, seeds, chunksize)`.
function seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, indices, index,
                             count = N) where {T,V,N}
    return _seed_zero_partials!(duals, x, structural_chunk(indices, index, count))
end

function _seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, idxs) where {T,V,N}
    seed = zero(Partials{N,V})
    return _seed!(duals, x, idxs) do value, _
        Dual{T,V,N}(value, seed)
    end
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, indices,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    return _seed!(duals, x, structural_chunk(indices, 1, N)) do value, i
        Dual{T,V,N}(value, seeds[i])
    end
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, indices, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    return _seed!(duals, x, structural_chunk(indices, index, chunksize)) do value, i
        Dual{T,V,N}(value, seeds[i])
    end
end
