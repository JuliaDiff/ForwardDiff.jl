###############
# API methods #
###############

"""
    ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Return `H(f)` evaluated at `x`, assuming `f` is called as `f(x)`.
The returned Hessian is exactly symmetric: its two triangles are filled from the same
derivative values.

Both axes of the result are indexed by the linear indices of `x`, so it is
`length(x)` by `length(x)`. For an `x` with structurally zero entries, such as a
`LowerTriangular`, `UpperTriangular` or `Diagonal` matrix, only the entries that are not
structurally zero are differentiated; the rows and columns of the others are zero. Note
that this makes the result of `hessian(f, ::Diagonal)` quadratic in `length(x)` and hence
quartic in the size of the diagonal — differentiate with respect to the diagonal vector
instead if that matters.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian(f::F, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, x), ::Val{CHK}=Val{true}()) where {F, T,CHK}
    require_one_based_indexing(x)
    CHK && checktag(T, f, x)
    H, _ = symmetric_hessian(f, x, cfg, nothing)
    return H
end

"""
    ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Compute `H(f)` evaluated at `x` and store the result(s) in `result`, assuming `f` is
called as `f(x)`. The stored Hessian is exactly symmetric: its two triangles are filled
from the same derivative values.

`result` has to hold `length(x)^2` entries, indexed as described for
`ForwardDiff.hessian`; a matrix `result` is written to as is and hence has to be
`length(x)` by `length(x)`.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::AbstractArray, f::F, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(result, x)
    CHK && checktag(T, f, x)
    H = reshape_hessian(result, x)
    symmetric_hessian!(H, f, x, cfg, nothing)
    return result
end

"""
    ForwardDiff.hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, result, x), check=Val{true}())

Exactly like `ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig)`,
but also stores the value and gradient in `result`. The default `cfg` is constructed as
`HessianConfig(f, result, x)`, though a config constructed as `HessianConfig(f, x)` may also
be used.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::DiffResult, f::F, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, result, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(x)
    CHK && checktag(T, f, x)
    H = reshape_hessian(result, x)
    _, ydual = symmetric_hessian!(H, f, x, cfg, DiffResults.gradient(result))
    result = DiffResults.value!(result, value(T, value(T, ydual)))
    return result
end

############################
# symmetric Hessian kernel #
############################

const HESSIAN_ERROR = DimensionMismatch("hessian(f, x) expects that f(x) is a real number. Perhaps you meant jacobian(f, x)?")

# Copy a block from the nested partials and fill its transpose. On diagonal blocks, read
# only the upper triangle so the result is exactly symmetric. `positions` maps a seeding position to
# the row and column of `H` it belongs to, i.e. to the linear index of `x` it was seeded from.
function extract_hessian_chunk!(::Type{T}, H, positions, ydual, roffset, coffset, rsize, csize) where {T}
    for r in 1:rsize
        i = positions[roffset + r]
        drow = partials(T, ydual, r)
        cstart = roffset == coffset ? r : 1
        for c in cstart:csize
            j = positions[coffset + c]
            h = partials(T, drow, c)
            H[i, j] = h
            H[j, i] = h
        end
    end
    return H
end

# The inner partials of a diagonal block contain the corresponding gradient chunk.
extract_hessian_gradient_chunk!(::Type{T}, ::Nothing, ydual, x, index, chunksize) where {T} = nothing
extract_hessian_gradient_chunk!(::Type{T}, grad, ydual, x, index, chunksize) where {T} =
    extract_gradient_chunk!(T, grad, value(T, ydual), x, index, chunksize)

function reshape_hessian(result::AbstractMatrix, x)
    size(result) == (length(x), length(x)) || throw(DimensionMismatch(
        lazy"cannot store the $(length(x))×$(length(x)) Hessian in a result of size $(size(result))"))
    return result
end
reshape_hessian(result::AbstractArray, x) = reshape(result, length(x), length(x))
reshape_hessian(result::DiffResult, x) = reshape_hessian(DiffResults.hessian(result), x)

# Evaluate one pair of chunks at a time using nested duals. Only one triangle of block
# pairs is evaluated; the other is filled by symmetry (see #836).
function symmetric_hessian_expr(result_definition::Expr)
    return quote
        xlen = structural_length(x)
        # Only the structurally non-zero entries of `x` are seeded, but both axes of the result are
        # indexed by the linear indices of `x`, as the columns of a Jacobian are. See #839.
        hlen = length(x)
        if xlen < N
            throw(ArgumentError(lazy"chunk size cannot be greater than ForwardDiff.structural_length(x) ($(N) > $(structural_length(x)))"))
        end

        # `N == 0` only for empty inputs, which still need one evaluation to determine the
        # output type and value.
        nblocks = xlen == 0 ? 1 : cld(xlen, N)

        xdual = cfg.gradient_config.duals
        iseeds = cfg.jacobian_config.seeds
        oseeds = cfg.gradient_config.seeds

        # The first evaluation determines the output type. Seeding the first block and clearing
        # the untouched tail partitions the fresh buffer, so every element is initialized once.
        seed_hessian_chunk!(xdual, x, 1, iseeds, oseeds)
        seed_hessian_chunk!(xdual, x, N + 1, nothing, nothing, xlen - N)
        ydual1 = f(xdual)
        ydual1 isa Real || throw(HESSIAN_ERROR)
        $(result_definition)
        # `H` is square and both its axes are indexed by the entries of `x`, so the columns a
        # Jacobian would receive derivatives in are also the rows and columns this sweep writes.
        positions = _indexable(structural_columns(H, x))
        # The entries that no block of the sweep writes belong to none of them in particular: for
        # `H` the rows and columns of the structurally zero entries of `x`, for `grad` their
        # entries. `value(T, ydual1)` is passed for its value type, which unlike `eltype` is a
        # number type even for a result that stores `Any`.
        zero_unseeded_columns!(T, H, value(T, ydual1), x)
        grad === nothing || zero_unseeded!(T, grad, value(T, ydual1), x)
        extract_hessian_chunk!(T, H, positions, ydual1, 0, 0, N, N)
        extract_hessian_gradient_chunk!(T, grad, ydual1, x, 1, N)
        nblocks > 1 && seed_hessian_chunk!(xdual, x, 1, nothing, nothing)

        for q in 2:nblocks
            qoffset = (q - 1) * N
            qsize = min(N, xlen - qoffset)
            # Off-diagonal blocks: p seeds columns and q seeds rows. The outer seeds for q
            # remain unchanged throughout this loop.
            seed_hessian_chunk!(xdual, x, qoffset + 1, nothing, oseeds, qsize)
            for p in 1:(q - 1)
                poffset = (p - 1) * N
                seed_hessian_chunk!(xdual, x, poffset + 1, iseeds, nothing)
                ydual = f(xdual)
                extract_hessian_chunk!(T, H, positions, ydual, qoffset, poffset, qsize, N)
                seed_hessian_chunk!(xdual, x, poffset + 1, nothing, nothing)
            end
            # The diagonal block adds q's inner seeds while retaining its outer seeds.
            seed_hessian_chunk!(xdual, x, qoffset + 1, iseeds, oseeds, qsize)
            ydual = f(xdual)
            extract_hessian_chunk!(T, H, positions, ydual, qoffset, qoffset, qsize, qsize)
            extract_hessian_gradient_chunk!(T, grad, ydual, x, qoffset + 1, qsize)
            seed_hessian_chunk!(xdual, x, qoffset + 1, nothing, nothing, qsize)
        end

        return H, ydual1
    end
end

@eval function symmetric_hessian(f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:(H = similar(x, valtype(T, valtype(T, typeof(ydual1))), hlen, hlen))))
end

@eval function symmetric_hessian!(H, f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:()))
end
