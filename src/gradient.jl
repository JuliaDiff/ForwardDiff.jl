###############
# API methods #
###############

const AllowedGradientConfig{F,H} = Union{GradientConfig{Tag{F,H}}, GradientConfig{Tag{Void,H}}}

gradient(f, x::AbstractArray, cfg::GradientConfig) = throw(ConfigHismatchError(f, cfg))
gradient!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::GradientConfig) = throw(ConfigHismatchError(f, cfg))

"""
    ForwardDiff.gradient(f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x))

Return `∇f` evaluated at `x`, assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Real)`.
"""
function gradient(f::F, x::AbstractArray, cfg::AllowedGradientConfig{F,H} = GradientConfig(f, x)) where {F,H}
    if chunksize(cfg) == length(x)
        return vector_mode_gradient(f, x, cfg)
    else
        return chunk_mode_gradient(f, x, cfg)
    end
end

"""
    ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x))

Compute `∇f` evaluated at `x` and store the result(s) in `result`, assuming `f` is called as
`f(x)`.

This method assumes that `isa(f(x), Real)`.
"""
function gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::AbstractArray, cfg::AllowedGradientConfig{F,H} = GradientConfig(f, x)) where {F,H}
    if chunksize(cfg) == length(x)
        vector_mode_gradient!(result, f, x, cfg)
    else
        chunk_mode_gradient!(result, f, x, cfg)
    end
    return result
end

@inline gradient(f::F, x::SArray) where {F} = vector_mode_gradient(f, x)
@inline gradient(f::F, x::SArray, cfg::AllowedGradientConfig{F,H}) where {F,H} = gradient(f, x)

@inline gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::SArray) where {F} = vector_mode_gradient!(result, f, x)
@inline gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::SArray, cfg::AllowedGradientConfig{F,H}) where {F,H} = gradient!(result, f, x)

#####################
# result extraction #
#####################

@generated function extract_gradient(y::Real, ::SArray{S,X,D,N}) where {S,X,D,N}
    result = Expr(:tuple, [:(partials(y, $i)) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        return SArray{S}($result)
    end
end

function extract_gradient!(result::DiffResult, y::Real)
    DiffBase.value!(result, y)
    grad = DiffBase.gradient(result)
    fill!(grad, zero(y))
    return result
end

function extract_gradient!(result::DiffResult, dual::Dual)
    result = DiffBase.value!(result, value(dual))
    result = DiffBase.gradient!(result, partials(dual))
    return result
end

extract_gradient!(result::AbstractArray, y::Real) = fill!(result, zero(y))
extract_gradient!(result::AbstractArray, dual::Dual) = copy!(result, partials(dual))

function extract_gradient_chunk!(result, dual, index, chunksize)
    offset = index - 1
    for i in 1:chunksize
        result[i + offset] = partials(dual, i)
    end
    return result
end

function extract_gradient_chunk!(result::DiffResult, dual, index, chunksize)
    extract_gradient_chunk!(DiffBase.gradient(result), dual, index, chunksize)
    return result
end

###############
# vector mode #
###############

function vector_mode_gradient(f::F, x, cfg) where {F}
    ydual = vector_mode_dual_eval(f, x, cfg)
    result = similar(x, valtype(ydual))
    return extract_gradient!(result, ydual)
end

function vector_mode_gradient!(result, f::F, x, cfg) where {F}
    ydual = vector_mode_dual_eval(f, x, cfg)
    extract_gradient!(result, ydual)
    return result
end

@inline function vector_mode_gradient(f::F, x::SArray) where F
    return extract_gradient(vector_mode_dual_eval(f, x), x)
end

@inline function vector_mode_gradient!(result, f::F, x::SArray) where F
    return extract_gradient!(result, vector_mode_dual_eval(f, x))
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
        extract_gradient_chunk!(result, ydual, 1, N)
        seed!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            ydual = f(xdual)
            extract_gradient_chunk!(result, ydual, i, N)
            seed!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        ydual = f(xdual)
        extract_gradient_chunk!(result, ydual, lastchunkindex, lastchunksize)

        # get the value, this is a no-op unless result is a DiffResult
        extract_value!(result, ydual)

        return result
    end
end

@eval function chunk_mode_gradient(f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:(result = similar(x, valtype(ydual)))))
end

@eval function chunk_mode_gradient!(result, f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:()))
end
