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
function hessian!(result::DiffResult, f::F, x::AbstractArray, cfg::HessianConfig{T,TO} = HessianConfig(f, result, x), ::Val{CHK}=Val{true}()) where {F,T,TO,CHK}
    require_one_based_indexing(x)
    CHK && checktag(T, f, x)
    _, ydual = symmetric_hessian!(reshape_hessian(result, x), f, x, cfg,
                                  DiffResults.gradient(result))
    result = DiffResults.value!(result, value(T, value(TO, ydual)))
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
function extract_hessian_chunk!(::Type{T}, ::Type{TO}, H, ydual::Dual{TO,<:Dual{T}}, indices, roffset, coffset, rsize, csize) where {T,TO}
    rows = structural_chunk(indices, roffset + 1, rsize)
    cols = structural_chunk(indices, coffset + 1, csize)
    for r in 1:rsize
        drow = partials(TO, ydual, r)
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

# Without both perturbations the block is zero.
function extract_hessian_chunk!(::Type{T}, ::Type{TO}, H, ydual, indices, roffset, coffset, rsize, csize) where {T,TO}
    rows = structural_chunk(indices, roffset + 1, rsize)
    cols = structural_chunk(indices, coffset + 1, csize)
    h = zero(valtype(T, valtype(TO, typeof(ydual))))
    for j in cols, i in rows
        H[i, j] = h
        H[j, i] = h
    end
    return H
end

# The inner partials of a diagonal block contain the corresponding gradient chunk.
# TODO: delegate to `extract_gradient_chunk!` once it takes its positions from `x` (#838).
extract_hessian_gradient_chunk!(::Type{T}, ::Type{TO}, ::Nothing, ydual, indices, index, chunksize) where {T,TO} = nothing
function extract_hessian_gradient_chunk!(::Type{T}, ::Type{TO}, grad, ydual, indices, index, chunksize) where {T,TO}
    dual = value(TO, ydual)
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
        Vout = valtype(T, valtype(TO, typeof(ydual1)))
        $(result_definition)
        # A second derivative needs both perturbations, a first derivative only the inner one:
        # what the result does not carry vanishes identically.
        zero_hessian = !(ydual1 isa Dual{TO,<:Dual{T}})
        zero_gradient = zero_hessian && !(ydual1 isa Dual{T})

        # Zero what no block writes: a derivative that vanishes, and the rows and columns of
        # the structural zeros of `x`, which are not variables.
        if zero_hessian || xlen != length(x)
            fill!(H, zero(Vout))
        end
        if grad !== nothing && (zero_gradient || xlen != length(x))
            fill!(grad, zero(Vout))
        end
        # off-diagonal blocks find second derivatives, diagonal ones also the gradient
        if zero_hessian && (zero_gradient || grad === nothing)
            return H, ydual1
        end
        extract_hessian_chunk!(T, TO, H, ydual1, indices, 0, 0, N, N)
        extract_hessian_gradient_chunk!(T, TO, grad, ydual1, indices, 1, N)
        if nblocks > 1
            seed_hessian_chunk!(xdual, x, indices, 1, nothing, nothing)
        end

        for q in 2:nblocks
            qoffset = (q - 1) * N
            qsize = min(N, xlen - qoffset)
            if !zero_hessian
                # Outer-i inner-j and outer-j inner-i round differently, so the outer layer always
                # takes the earlier position -- else the result would depend on the chunk size.
                # q's inner seeds remain unchanged throughout this loop.
                seed_hessian_chunk!(xdual, x, indices, qoffset + 1, iseeds, nothing, qsize)
                for p in 1:(q - 1)
                    poffset = (p - 1) * N
                    seed_hessian_chunk!(xdual, x, indices, poffset + 1, nothing, oseeds)
                    ydual = f(xdual)
                    extract_hessian_chunk!(T, TO, H, ydual, indices, poffset, qoffset, N, qsize)
                    seed_hessian_chunk!(xdual, x, indices, poffset + 1, nothing, nothing)
                end
            end
            # The diagonal block adds q's outer seeds while retaining its inner seeds.
            seed_hessian_chunk!(xdual, x, indices, qoffset + 1, iseeds, oseeds, qsize)
            ydual = f(xdual)
            extract_hessian_chunk!(T, TO, H, ydual, indices, qoffset, qoffset, qsize, qsize)
            extract_hessian_gradient_chunk!(T, TO, grad, ydual, indices, qoffset + 1, qsize)
            seed_hessian_chunk!(xdual, x, indices, qoffset + 1, nothing, nothing, qsize)
        end

        return H, ydual1
    end
end

@eval function symmetric_hessian(f::F, x, cfg::HessianConfig{T,TO,V,N}, grad) where {F,T,TO,V,N}
    $(symmetric_hessian_expr(:(H = similar(x, Vout, length(x), length(x)))))
end

@eval function symmetric_hessian!(H, f::F, x, cfg::HessianConfig{T,TO,V,N}, grad) where {F,T,TO,V,N}
    $(symmetric_hessian_expr(:()))
end
