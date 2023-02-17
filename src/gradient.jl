###############
# API methods #
###############

"""
    ForwardDiff.gradient(f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x), check=Val{true}())

Return `∇f` evaluated at `x`, assuming `f` is called as `f(x)`.
The array `∇f` has the same shape as `x`, and its elements are
`∇f[j, k, ...] = ∂f/∂x[j, k, ...]`.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function gradient(f, x::AbstractArray, cfg::GradientConfig{T} = GradientConfig(f, x), ::Val{CHK}=Val{true}()) where {T, CHK}
    CHK && checktag(T, f, x)
    if chunksize(cfg) == length(x)
        return vector_mode_gradient(f, x, cfg)
    else
        return chunk_mode_gradient(f, x, cfg)
    end
end

"""
    ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x), check=Val{true}())

Compute `∇f` evaluated at `x` and store the result(s) in `result`, assuming `f` is called as
`f(x)`.

This method assumes that `isa(f(x), Real)`.

"""
function gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::AbstractArray, cfg::GradientConfig{T} = GradientConfig(f, x), ::Val{CHK}=Val{true}()) where {T, CHK, F}
    CHK && checktag(T, f, x)
    if chunksize(cfg) == length(x)
        vector_mode_gradient!(result, f, x, cfg)
    else
        chunk_mode_gradient!(result, f, x, cfg)
    end
    return result
end

gradient(f, x::Real) = throw(DimensionMismatch("gradient(f, x) expects that x is an array. Perhaps you meant derivative(f, x)?"))

#####################
# result extraction #
#####################

function extract_gradient!(::Type{T}, result::DiffResult, y::Real) where {T}
    result = DiffResults.value!(result, y)
    grad = DiffResults.gradient(result)
    fill!(grad, zero(y))
    return result
end

function extract_gradient!(::Type{T}, result::DiffResult, dual::Dual) where {T}
    result = DiffResults.value!(result, value(T, dual))
    result = DiffResults.gradient!(result, partials(T, dual))
    return result
end

extract_gradient!(::Type{T}, result::AbstractArray, y::Real) where {T} = fill!(result, zero(y))
extract_gradient!(::Type{T}, result::AbstractArray, dual::Dual) where {T}= copyto!(result, partials(T, dual))

function extract_gradient_chunk!(::Type{T}, result, dual, index, chunksize) where {T}
    offset = index - 1
    for i in 1:chunksize
        result[i + offset] = partials(T, dual, i)
    end
    return result
end

function extract_gradient_chunk!(::Type{T}, result::DiffResult, dual, index, chunksize) where {T}
    extract_gradient_chunk!(T, DiffResults.gradient(result), dual, index, chunksize)
    return result
end

extract_gradient_chunk!(::Type, result, dual::AbstractArray, index, chunksize) = throw(GRAD_ERROR)
extract_gradient_chunk!(::Type, result::DiffResult, dual::AbstractArray, index, chunksize) = throw(GRAD_ERROR)

const GRAD_ERROR = DimensionMismatch("gradient(f, x) expects that f(x) is a real number. Perhaps you meant jacobian(f, x)?")

###############
# vector mode #
###############

function vector_mode_gradient(f::F, x, cfg::GradientConfig{T}) where {T, F}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa Real || throw(GRAD_ERROR)
    result = similar(x, valtype(ydual))
    return extract_gradient!(T, result, ydual)
end

function vector_mode_gradient!(result, f::F, x, cfg::GradientConfig{T}) where {T, F}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    result = extract_gradient!(T, result, ydual)
    return result
end

##############
# chunk mode #
##############

function chunk_mode_gradient_expr(result_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # seed work vectors
        xdual = cfg.duals
        seeds = cfg.seeds
        seed!(xdual, x)

        # do first chunk manually to calculate output type
        seed!(xdual, x, 1, seeds)
        ydual = f(xdual)
        $(result_definition)
        extract_gradient_chunk!(T, result, ydual, 1, N)
        seed!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            ydual = f(xdual)
            extract_gradient_chunk!(T, result, ydual, i, N)
            seed!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        ydual = f(xdual)
        extract_gradient_chunk!(T, result, ydual, lastchunkindex, lastchunksize)

        # get the value, this is a no-op unless result is a DiffResult
        extract_value!(T, result, ydual)

        return result
    end
end

@eval function chunk_mode_gradient(f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:(result = similar(x, valtype(ydual)))))
end

@eval function chunk_mode_gradient!(result, f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:()))
end
