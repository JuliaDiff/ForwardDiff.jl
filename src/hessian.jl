###############
# API methods #
###############

"""
    ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Return `H(f)` evaluated at `x`, assuming `f` is called as `f(x)`.

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

Compute `H(f)` (i.e. `J(∇(f))`) evaluated at `x` and store the result(s) in `result`,
assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::AbstractArray, f::F, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(result, x)
    CHK && checktag(T, f, x)
    xlen = structural_length(x)
    H = result isa AbstractMatrix && size(result) == (xlen, xlen) ? result : reshape(result, xlen, xlen)
    symmetric_hessian!(H, f, x, cfg, nothing)
    return result
end

"""
    ForwardDiff.hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, result, x), check=Val{true}())

Exactly like `ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig)`, but
because `isa(result, DiffResult)`, `cfg` is constructed as `HessianConfig(f, result, x)` instead of
`HessianConfig(f, x)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::DiffResult, f::F, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, result, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(x)
    CHK && checktag(T, f, x)
    xlen = structural_length(x)
    hess = DiffResults.hessian(result)
    H = hess isa AbstractMatrix && size(hess) == (xlen, xlen) ? hess : reshape(hess, xlen, xlen)
    _, ydual = symmetric_hessian!(H, f, x, cfg, DiffResults.gradient(result))
    result = DiffResults.value!(result, value(T, value(T, ydual)))
    return result
end

############################
# symmetric Hessian kernel #
############################

const HESSIAN_ERROR = DimensionMismatch("hessian(f, x) expects that f(x) is a real number. Perhaps you meant jacobian(f, x)?")

# Seed a chunk in either layer of the nested duals. A `nothing` seed clears that layer.
function seed_hessian_chunk!(duals::AbstractArray{Dual{T,Dual{T,V,N},N}}, x, index,
                             iseeds::Union{Nothing,NTuple{N,Partials{N,V}}},
                             oseeds::Union{Nothing,NTuple{N,Partials{N,Dual{T,V,N}}}},
                             chunksize = N) where {T,V,N}
    izero = zero(Partials{N,V})
    ozero = zero(Partials{N,Dual{T,V,N}})
    idxs = Iterators.drop(structural_eachindex(duals, x), index - 1)
    if isbitstype(V)
        for (i, idx) in zip(1:chunksize, idxs)
            inner = Dual{T,V,N}(x[idx], iseeds === nothing ? izero : iseeds[i])
            duals[idx] = Dual{T,Dual{T,V,N},N}(inner, oseeds === nothing ? ozero : oseeds[i])
        end
    else
        for (i, idx) in zip(1:chunksize, idxs)
            if isassigned(x, idx)
                inner = Dual{T,V,N}(x[idx], iseeds === nothing ? izero : iseeds[i])
                duals[idx] = Dual{T,Dual{T,V,N},N}(inner, oseeds === nothing ? ozero : oseeds[i])
            else
                Base._unsetindex!(duals, idx)
            end
        end
    end
    return duals
end

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

        nblocks = xlen == 0 ? 1 : div(xlen + N - 1, N)

        xdual = cfg.gradient_config.duals
        iseeds = cfg.jacobian_config.seeds
        oseeds = cfg.gradient_config.seeds

        # Keep all unseeded blocks at zero between evaluations.
        seed_hessian_chunk!(xdual, x, 1, nothing, nothing, xlen)

        # The first evaluation determines the output type.
        seed_hessian_chunk!(xdual, x, 1, iseeds, oseeds)
        ydual1 = f(xdual)
        ydual1 isa Real || throw(HESSIAN_ERROR)
        $(result_definition)
        extract_hessian_chunk!(T, H, ydual1, 0, 0, N, N)
        extract_hessian_gradient_chunk!(T, grad, ydual1, 1, N)
        seed_hessian_chunk!(xdual, x, 1, nothing, nothing)

        for q in 2:nblocks
            qoffset = (q - 1) * N
            qsize = min(N, xlen - qoffset)
            # Off-diagonal blocks: p seeds columns and q seeds rows.
            for p in 1:(q - 1)
                poffset = (p - 1) * N
                seed_hessian_chunk!(xdual, x, poffset + 1, iseeds, nothing)
                seed_hessian_chunk!(xdual, x, qoffset + 1, nothing, oseeds, qsize)
                ydual = f(xdual)
                extract_hessian_chunk!(T, H, ydual, qoffset, poffset, qsize, N)
                seed_hessian_chunk!(xdual, x, poffset + 1, nothing, nothing)
                seed_hessian_chunk!(xdual, x, qoffset + 1, nothing, nothing, qsize)
            end
            # Diagonal blocks seed both layers.
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
    $(symmetric_hessian_expr(:(H = similar(x, typeof(value(T, value(T, ydual1))), xlen, xlen))))
end

@eval function symmetric_hessian!(H, f::F, x, cfg::HessianConfig{T,V,N}, grad) where {F,T,V,N}
    $(symmetric_hessian_expr(:()))
end
