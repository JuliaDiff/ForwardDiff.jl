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

# The set of linear positions an array stores, so that two arrays can be compared without comparing
# their positions themselves. `structural_indices` below enumerates the positions of each kind.
abstract type StructuralKind end
struct AllEntries <: StructuralKind end
struct LowerTriangle <: StructuralKind end
struct UpperTriangle <: StructuralKind end
struct MainDiagonal <: StructuralKind end

structural_kind(::AbstractArray) = AllEntries()
structural_kind(::LowerTriangular) = LowerTriangle()
structural_kind(::UpperTriangular) = UpperTriangle()
structural_kind(::Diagonal) = MainDiagonal()

# The linear indices of the entries that are seeded, in seeding order. Being linear indices of the
# array itself, one position indexes the input, the work buffer and the result alike, and is a
# Jacobian column number as it stands. Configs hold one per work buffer, so that a sweep indexes
# straight to a chunk.
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

# The positions of the `count` entries starting at structural position `index`. Allocation-free, and
# a window overrunning the end is a `BoundsError` rather than a silently truncated chunk.
structural_chunk(indices, index, count) = view(indices, index:(index + count - 1))

# Does an array of kind `outer` store every position of an array of kind `inner`? Add a method here
# whenever a kind is added above.
structural_issubset(inner::StructuralKind, outer::StructuralKind) = inner === outer
structural_issubset(::StructuralKind, ::AllEntries) = true
structural_issubset(::MainDiagonal, ::LowerTriangle) = true
structural_issubset(::MainDiagonal, ::UpperTriangle) = true

# Checks that the structural positions of `x` are positions of `y` as well. Being linear indices,
# they only constrain how many entries `y` has, not its shape.
function check_structural_indices(x::AbstractArray, y::AbstractArray)
    require_one_based_indexing(y)
    structural_issubset(structural_kind(x), structural_kind(y)) || throw(ArgumentError(LazyString(
        "an array of type ", nameof(typeof(y)), " does not store every entry of an array of type ",
        nameof(typeof(x)), ": the two are structured differently")))
    length(x) == length(y) || throw(DimensionMismatch(
        lazy"expected an array with $(length(x)) elements, got an array with $(length(y)) elements"))
    return nothing
end

# The config's positions were built for its work buffer, so they fit `x` exactly when the two are
# structurally interchangeable. The kind comparison is a compile-time constant, hence free per call.
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

# Mirrors an unassigned entry of `x` into the work buffer. `Base._unsetindex!` is implemented for
# `Array` alone, a structured wrapper being a view onto a parent with no slot of its own to unset.
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

function seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, idxs) where {T,V,N}
    seed = zero(Partials{N,V})
    return _seed!(duals, x, idxs) do value, _
        Dual{T,V,N}(value, seed)
    end
end

seed_zero_partials!(duals::AbstractArray{Dual{T,V,N}}, x, indices, index, count = N) where {T,V,N} =
    seed_zero_partials!(duals, x, structural_chunk(indices, index, count))

seed!(duals::AbstractArray{Dual{T,V,N}}, x, indices,
      seeds::NTuple{N,Partials{N,V}}) where {T,V,N} = seed!(duals, x, indices, 1, seeds)

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, indices, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    return _seed!(duals, x, structural_chunk(indices, index, chunksize)) do value, i
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
    return _seed!(duals, x, structural_chunk(indices, index, chunksize)) do value, i
        inner = Dual{T,V,N}(value, iseeds === nothing ? izero : iseeds[i])
        Dual{T,Dual{T,V,N},N}(inner, oseeds === nothing ? ozero : oseeds[i])
    end
end
