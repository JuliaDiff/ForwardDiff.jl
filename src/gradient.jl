###############
# API methods #
###############

const AllowedGradientConfig{F,H} = Union{GradientConfig{Tag{F,H}}, GradientConfig{Tag{Void,H}}}

gradient(f, x, cfg::GradientConfig) = throw(ConfigHismatchError(f, cfg))
gradient!(out, f, x, cfg::GradientConfig) = throw(ConfigHismatchError(f, cfg))

function gradient(f::F, x, cfg::AllowedGradientConfig{F,H} = GradientConfig(f, x)) where {F,H}
    if chunksize(cfg) == length(x)
        return vector_mode_gradient(f, x, cfg)
    else
        return chunk_mode_gradient(f, x, cfg)
    end
end

function gradient!(out, f::F, x, cfg::AllowedGradientConfig{F,H} = GradientConfig(f, x)) where {F,H}
    if chunksize(cfg) == length(x)
        vector_mode_gradient!(out, f, x, cfg)
    else
        chunk_mode_gradient!(out, f, x, cfg)
    end
    return out
end

@inline gradient(f::F, x::SArray) where {F} = vector_mode_gradient(f, x)

@inline gradient!(out, f::F, x::SArray) where {F} = vector_mode_gradient!(out, f, x)

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

function extract_gradient!(out::DiffResult, y::Real)
    DiffBase.value!(out, y)
    grad = DiffBase.gradient(out)
    fill!(grad, zero(y))
    return out
end

function extract_gradient!(out::DiffResult, dual::Dual)
    out = DiffBase.value!(out, value(dual))
    out = DiffBase.gradient!(out, partials(dual))
    return out
end

extract_gradient!(out::AbstractArray, y::Real) = fill!(out, zero(y))
extract_gradient!(out::AbstractArray, dual::Dual) = copy!(out, partials(dual))

function extract_gradient_chunk!(out, dual, index, chunksize)
    offset = index - 1
    for i in 1:chunksize
        out[i + offset] = partials(dual, i)
    end
    return out
end

function extract_gradient_chunk!(out::DiffResult, dual, index, chunksize)
    extract_gradient_chunk!(DiffBase.gradient(out), dual, index, chunksize)
    return out
end

###############
# vector mode #
###############

function vector_mode_gradient(f::F, x, cfg) where {F}
    ydual = vector_mode_dual_eval(f, x, cfg)
    out = similar(x, valtype(ydual))
    return extract_gradient!(out, ydual)
end

function vector_mode_gradient!(out, f::F, x, cfg) where {F}
    ydual = vector_mode_dual_eval(f, x, cfg)
    extract_gradient!(out, ydual)
    return out
end

@inline function vector_mode_gradient(f::F, x::SArray) where F
    return extract_gradient(vector_mode_dual_eval(f, x), x)
end

@inline function vector_mode_gradient!(out, f::F, x::SArray) where F
    return extract_gradient!(out, vector_mode_dual_eval(f, x))
end

##############
# chunk mode #
##############

function chunk_mode_gradient_expr(out_definition::Expr)
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
        $(out_definition)
        extract_gradient_chunk!(out, ydual, 1, N)
        seed!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            ydual = f(xdual)
            extract_gradient_chunk!(out, ydual, i, N)
            seed!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        ydual = f(xdual)
        extract_gradient_chunk!(out, ydual, lastchunkindex, lastchunksize)

        # get the value, this is a no-op unless out is a DiffResult
        extract_value!(out, ydual)

        return out
    end
end

@eval function chunk_mode_gradient(f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:(out = similar(x, valtype(ydual)))))
end

@eval function chunk_mode_gradient!(out, f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:()))
end
