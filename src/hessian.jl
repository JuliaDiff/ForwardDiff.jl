###############
# API methods #
###############

"""
    ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Return `H(f)` evaluated at `x`, assuming `f` is called as `f(x)`.
Multidimensional arrays are flattened in iteration order: the array `H(f)` has shape
`length(x) × length(x)`, and its elements are `H(f)[j,k] = ∂²f(x)/∂x[j]∂x[k]`.
The returned Hessian is exactly symmetric: its two triangles are filled from the same
derivative values.

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

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::AbstractArray, f::F, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(result, x)
    CHK && checktag(T, f, x)
    symmetric_hessian!(reshape_hessian(result, x), f, x, cfg, nothing)
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
    _, ydual = symmetric_hessian!(reshape_hessian(result, x), f, x, cfg,
                                  DiffResults.gradient(result))
    result = DiffResults.value!(result, value(T, value(T, ydual)))
    return result
end

############################
# symmetric Hessian kernel #
############################

const HESSIAN_ERROR = DimensionMismatch("hessian(f, x) expects that f(x) is a real number. Perhaps you meant jacobian(f, x)?")

# Mirrors `reshape_jacobian`. The sweep writes the result entry by entry, so nothing else checks it.
function reshape_hessian(result::AbstractMatrix, x)
    require_one_based_indexing(result)
    if size(result) != (length(x), length(x))
        throw(DimensionMismatch(lazy"cannot store the $(length(x))×$(length(x)) Hessian in a result of size $(size(result))"))
    end
    return result
end
function reshape_hessian(result::AbstractArray, x)
    require_one_based_indexing(result)
    if length(result) != length(x)^2
        throw(DimensionMismatch(lazy"cannot store the $(length(x))×$(length(x)) Hessian in a result of length $(length(result))"))
    end
    return reshape(result, length(x), length(x))
end
function reshape_hessian(result::DiffResult, x)
    structural_eachindex(DiffResults.gradient(result), x)
    return reshape_hessian(DiffResults.hessian(result), x)
end

# Copy a block from the nested partials and fill its transpose. On diagonal blocks, read
# only the upper triangle so the result is exactly symmetric. `indices` maps a block
# position to its row and column, both being linear indices of `x`.
function extract_hessian_chunk!(::Type{T}, H, ydual, indices, roffset, coffset, rsize, csize) where {T}
    rows = structural_chunk(indices, roffset + 1, rsize)
    cols = structural_chunk(indices, coffset + 1, csize)
    for r in 1:rsize
        drow = partials(T, ydual, r)
        i = rows[r]
        cstart = roffset == coffset ? r : 1
        for c in cstart:csize
            h = partials(T, drow, c)
            j = cols[c]
            H[i, j] = h
            H[j, i] = h
        end
    end
    return H
end

# The inner partials of a diagonal block contain the corresponding gradient chunk.
# TODO: delegate to `extract_gradient_chunk!` once it takes its positions from `x` (#838).
extract_hessian_gradient_chunk!(::Type{T}, ::Nothing, ydual, indices, index, chunksize) where {T} = nothing
function extract_hessian_gradient_chunk!(::Type{T}, grad, ydual, indices, index, chunksize) where {T}
    dual = value(T, ydual)
    for (i, idx) in enumerate(structural_chunk(indices, index, chunksize))
        grad[idx] = partials(T, dual, i)
    end
    return grad
end

# Evaluate one pair of chunks at a time using nested duals. Only one triangle of block
# pairs is evaluated; the other is filled by symmetry (see #836).
function symmetric_hessian_expr(result_definition::Expr)
    return quote
        xlen = structural_length(x)
        if xlen < N
            throw(ArgumentError(lazy"chunk size cannot be greater than the number of differentiated entries of x ($(N) > $(xlen))"))
        end

        # `N == 0` only for empty inputs, which still need one evaluation to determine the
        # output type and value.
        nblocks = xlen == 0 ? 1 : cld(xlen, N)

        xdual = cfg.duals
        iseeds = cfg.iseeds
        oseeds = cfg.oseeds
        indices = structural_linearindices(xdual, x)

        # The first evaluation determines the output type. Seeding the first block and clearing
        # the untouched tail partitions the fresh buffer, so every element is initialized once.
        seed_hessian_chunk!(xdual, x, indices, 1, iseeds, oseeds)
        seed_hessian_chunk!(xdual, x, indices, N + 1, nothing, nothing, xlen - N)
        ydual1 = f(xdual)
        ydual1 isa Real || throw(HESSIAN_ERROR)
        $(result_definition)
        # the structural zeros of `x` are not variables, so no block writes their rows and columns
        if xlen != length(x)
            fill!(H, zero(eltype(H)))
            if grad !== nothing
                fill!(grad, zero(eltype(grad)))
            end
        end
        extract_hessian_chunk!(T, H, ydual1, indices, 0, 0, N, N)
        extract_hessian_gradient_chunk!(T, grad, ydual1, indices, 1, N)
        if nblocks > 1
            seed_hessian_chunk!(xdual, x, indices, 1, nothing, nothing)
        end

        for q in 2:nblocks
            qoffset = (q - 1) * N
            qsize = min(N, xlen - qoffset)
            # Outer-i inner-j and outer-j inner-i round differently, so the outer layer always
            # takes the earlier position -- else the result would depend on the chunk size.
            # q's inner seeds remain unchanged throughout this loop.
            seed_hessian_chunk!(xdual, x, indices, qoffset + 1, iseeds, nothing, qsize)
            for p in 1:(q - 1)
                poffset = (p - 1) * N
                seed_hessian_chunk!(xdual, x, indices, poffset + 1, nothing, oseeds)
                ydual = f(xdual)
                extract_hessian_chunk!(T, H, ydual, indices, poffset, qoffset, N, qsize)
                seed_hessian_chunk!(xdual, x, indices, poffset + 1, nothing, nothing)
            end
            # The diagonal block adds q's outer seeds while retaining its inner seeds.
            seed_hessian_chunk!(xdual, x, indices, qoffset + 1, iseeds, oseeds, qsize)
            ydual = f(xdual)
            extract_hessian_chunk!(T, H, ydual, indices, qoffset, qoffset, qsize, qsize)
            extract_hessian_gradient_chunk!(T, grad, ydual, indices, qoffset + 1, qsize)
            seed_hessian_chunk!(xdual, x, indices, qoffset + 1, nothing, nothing, qsize)
        end

        return H, ydual1
    end
end

@eval function symmetric_hessian(f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:(H = similar(x, valtype(T, valtype(T, typeof(ydual1))), length(x), length(x)))))
end

@eval function symmetric_hessian!(H, f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:()))
end
