###############
# API methods #
###############

"""
    ForwardDiff.jacobian(f, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f, x), check=Val{true}())

Return `J(f)` evaluated at `x`, assuming `f` is called as `f(x)`.
Multidimensional arrays are flattened in iteration order: the array
`J(f)` has shape `length(f(x)) × length(x)`, and its elements are
`J(f)[j,k] = ∂f(x)[j]/∂x[k]`.  When `x` is a vector, this means
that `jacobian(x->[f(x)], x)` is the transpose of `gradient(f, x)`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian(f, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f, x), ::Val{CHK}=Val{true}()) where {T,CHK}
    CHK && checktag(T, f, x)
    if chunksize(cfg) == length(x)
        return vector_mode_jacobian(f, x, cfg)
    else
        return chunk_mode_jacobian(f, x, cfg)
    end
end

"""
    ForwardDiff.jacobian(f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f!, y, x), check=Val{true}())

Return `J(f!)` evaluated at `x`,  assuming `f!` is called as `f!(y, x)` where the result is
stored in `y`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian(f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {T, CHK}
    CHK && checktag(T, f!, x)
    if chunksize(cfg) == length(x)
        return vector_mode_jacobian(f!, y, x, cfg)
    else
        return chunk_mode_jacobian(f!, y, x, cfg)
    end
end


"""
    ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f, x), check=Val{true}())

Compute `J(f)` evaluated at `x` and store the result(s) in `result`, assuming `f` is called
as `f(x)`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f, x), ::Val{CHK}=Val{true}()) where {T, CHK}
    CHK && checktag(T, f, x)
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(result, f, x, cfg)
    else
        chunk_mode_jacobian!(result, f, x, cfg)
    end
    return result
end

"""
    ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f!, y, x), check=Val{true}())

Compute `J(f!)` evaluated at `x` and store the result(s) in `result`, assuming `f!` is
called as `f!(y, x)` where the result is stored in `y`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {T,CHK}
    CHK && checktag(T, f!, x)
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(result, f!, y, x, cfg)
    else
        chunk_mode_jacobian!(result, f!, y, x, cfg)
    end
    return result
end

jacobian(f, x::Real) = throw(DimensionMismatch("jacobian(f, x) expects that x is an array. Perhaps you meant derivative(f, x)?"))

#####################
# result extraction #
#####################

function extract_jacobian!(::Type{T}, result::AbstractArray, ydual::AbstractArray, n) where {T}
    out_reshaped = reshape(result, length(ydual), n)
    ydual_reshaped = vec(ydual)
    # Use closure to avoid GPU broadcasting with Type
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    out_reshaped .= partials_wrap.(ydual_reshaped, transpose(1:n))
    return result
end

function extract_jacobian!(::Type{T}, result::MutableDiffResult, ydual::AbstractArray, n) where {T}
    extract_jacobian!(T, DiffResults.jacobian(result), ydual, n)
    return result
end

function extract_jacobian_chunk!(::Type{T}, result, ydual, index, chunksize) where {T}
    ydual_reshaped = vec(ydual)
    offset = index - 1
    irange = 1:chunksize
    col = irange .+ offset
    # Use closure to avoid GPU broadcasting with Type
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    result[:, col] .= partials_wrap.(ydual_reshaped, transpose(irange))
    return result
end

reshape_jacobian(result, ydual, xdual) = reshape(result, length(ydual), length(xdual))
reshape_jacobian(result::DiffResult, ydual, xdual) = reshape_jacobian(DiffResults.jacobian(result), ydual, xdual)

###############
# vector mode #
###############

function vector_mode_jacobian(f::F, x, cfg::JacobianConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa AbstractArray || throw(JACOBIAN_ERROR)
    result = similar(ydual, valtype(eltype(ydual)), length(ydual), N)
    extract_jacobian!(T, result, ydual, N)
    extract_value!(T, result, ydual)
    return result
end

function vector_mode_jacobian(f!::F, y, x, cfg::JacobianConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    result = similar(y, length(y), N)
    extract_jacobian!(T, result, ydual, N)
    map!(d -> value(T,d), y, ydual)
    return result
end

function vector_mode_jacobian!(result, f::F, x, cfg::JacobianConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f, cfg, x)
    extract_jacobian!(T, result, ydual, N)
    extract_value!(T, result, ydual)
    return result
end

function vector_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T}) where {F,T}
    N = chunksize(cfg)
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    extract_jacobian!(T, result, ydual, N)
    extract_value!(T, result, y, ydual)
    return result
end

const JACOBIAN_ERROR = DimensionMismatch("jacobian(f, x) expects that f(x) is an array. Perhaps you meant gradient(f, x)?")

# chunk mode #
#------------#

function jacobian_chunk_mode_expr(work_array_definition::Expr, compute_ydual::Expr,
                                  result_definition::Expr, y_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # seed work arrays
        $(work_array_definition)
        seeds = cfg.seeds

        # do first chunk manually to calculate output type
        seed!(xdual, x, 1, seeds)
        $(compute_ydual)
        ydual isa AbstractArray || throw(JACOBIAN_ERROR)
        $(result_definition)
        out_reshaped = reshape_jacobian(result, ydual, xdual)
        extract_jacobian_chunk!(T, out_reshaped, ydual, 1, N)
        seed!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            $(compute_ydual)
            extract_jacobian_chunk!(T, out_reshaped, ydual, i, N)
            seed!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        $(compute_ydual)
        extract_jacobian_chunk!(T, out_reshaped, ydual, lastchunkindex, lastchunksize)

        $(y_definition)

        return result
    end
end

@eval function chunk_mode_jacobian(f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(result = similar(ydual, valtype(eltype(ydual)), length(ydual), xlen)),
                               :()))
end

@eval function chunk_mode_jacobian(f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(result = similar(y, length(y), xlen)),
                               :(map!(d -> value(T,d), y, ydual))))
end

@eval function chunk_mode_jacobian!(result, f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(),
                               :(extract_value!(T, result, ydual))))
end

@eval function chunk_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(),
                               :(extract_value!(T, result, y, ydual))))
end
