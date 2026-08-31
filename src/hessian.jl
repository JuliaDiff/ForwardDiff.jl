###############
# API methods #
###############

"""
    ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Return `H(f)` evaluated at `x`, assuming `f` is called as `f(x)`.
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
    xlen = structural_length(x)
    H = result isa AbstractMatrix ? result : reshape(result, xlen, xlen)
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
    xlen = structural_length(x)
    hess = DiffResults.hessian(result)
    H = hess isa AbstractMatrix ? hess : reshape(hess, xlen, xlen)
    _, ydual = symmetric_hessian!(H, f, x, cfg, DiffResults.gradient(result))
    result = DiffResults.value!(result, value(T, value(T, ydual)))
    return result
end

############################
# symmetric Hessian kernel #
############################

const HESSIAN_ERROR = DimensionMismatch("hessian(f, x) expects that f(x) is a real number. Perhaps you meant jacobian(f, x)?")

# Copy a block from the nested partials and fill its transpose. On diagonal blocks, read
# only the upper triangle so the result is exactly symmetric.
function extract_hessian_chunk!(::Type{T}, H, ydual, roffset, coffset, rsize, csize) where {T}
    for r in 1:rsize
        drow = partials(T, ydual, r)
        cstart = roffset == coffset ? r : 1
        for c in cstart:csize
            h = partials(T, drow, c)
            H[roffset + r, coffset + c] = h
            H[coffset + c, roffset + r] = h
        end
    end
    return H
end

# The inner partials of a diagonal block contain the corresponding gradient chunk.
extract_hessian_gradient_chunk!(::Type{T}, ::Nothing, ydual, index, chunksize) where {T} = nothing
extract_hessian_gradient_chunk!(::Type{T}, grad, ydual, index, chunksize) where {T} =
    extract_gradient_chunk!(T, grad, value(T, ydual), index, chunksize)

# Evaluate one pair of chunks at a time using nested duals. Only one triangle of block
# pairs is evaluated; the other is filled by symmetry (see #836).
function symmetric_hessian_expr(result_definition::Expr)
    return quote
        xlen = structural_length(x)
        if xlen < N
            throw(ArgumentError(lazy"chunk size cannot be greater than ForwardDiff.structural_length(x) ($(N) > $(structural_length(x)))"))
        end

        # `N == 0` only for empty inputs, which still need one evaluation to determine the
        # output type and value.
        nblocks = xlen == 0 ? 1 : cld(xlen, N)

        xdual = cfg.duals
        iseeds = cfg.iseeds
        oseeds = cfg.oseeds

        # The first evaluation determines the output type. Seeding the first block and clearing
        # the untouched tail partitions the fresh buffer, so every element is initialized once.
        seed_hessian_chunk!(xdual, x, 1, iseeds, oseeds)
        seed_hessian_chunk!(xdual, x, N + 1, nothing, nothing, xlen - N)
        ydual1 = f(xdual)
        ydual1 isa Real || throw(HESSIAN_ERROR)
        $(result_definition)
        extract_hessian_chunk!(T, H, ydual1, 0, 0, N, N)
        extract_hessian_gradient_chunk!(T, grad, ydual1, 1, N)
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
                extract_hessian_chunk!(T, H, ydual, qoffset, poffset, qsize, N)
                seed_hessian_chunk!(xdual, x, poffset + 1, nothing, nothing)
            end
            # The diagonal block adds q's inner seeds while retaining its outer seeds.
            seed_hessian_chunk!(xdual, x, qoffset + 1, iseeds, oseeds, qsize)
            ydual = f(xdual)
            extract_hessian_chunk!(T, H, ydual, qoffset, qoffset, qsize, qsize)
            extract_hessian_gradient_chunk!(T, grad, ydual, qoffset + 1, qsize)
            seed_hessian_chunk!(xdual, x, qoffset + 1, nothing, nothing, qsize)
        end

        return H, ydual1
    end
end

@eval function symmetric_hessian(f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:(H = similar(x, valtype(T, valtype(T, typeof(ydual1))), xlen, xlen))))
end

@eval function symmetric_hessian!(H, f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:()))
end
