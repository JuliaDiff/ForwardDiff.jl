###############
# API methods #
###############

const AllowedJacobianConfig{F,H} = Union{JacobianConfig{Tag{F,H}}, JacobianConfig{Tag{Void,H}}}

jacobian(f, x, cfg::JacobianConfig) = throw(ConfigMismatchError(f, cfg))
jacobian(f!, y, x, cfg::JacobianConfig) = throw(ConfigMismatchError(f!, cfg))
jacobian!(out, f, x, cfg::JacobianConfig) = throw(ConfigMismatchError(f, cfg))
jacobian!(out, f!, y, x, cfg::JacobianConfig) = throw(ConfigMismatchError(f!, cfg))

function jacobian(f::F, x, cfg::AllowedJacobianConfig{F,H} = JacobianConfig(f, x)) where {F,H}
    if chunksize(cfg) == length(x)
        return vector_mode_jacobian(f, x, cfg)
    else
        return chunk_mode_jacobian(f, x, cfg)
    end
end

function jacobian(f!::F, y, x, cfg::AllowedJacobianConfig{F,H} = JacobianConfig(f!, y, x)) where {F,H}
    if chunksize(cfg) == length(x)
        return vector_mode_jacobian(f!, y, x, cfg)
    else
        return chunk_mode_jacobian(f!, y, x, cfg)
    end
end

function jacobian!(out, f::F, x, cfg::AllowedJacobianConfig{F,H} = JacobianConfig(f, x)) where {F,H}
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(out, f, x, cfg)
    else
        chunk_mode_jacobian!(out, f, x, cfg)
    end
    return out
end

function jacobian!(out, f!::F, y, x, cfg::AllowedJacobianConfig{F,H} = JacobianConfig(f!, y, x)) where {F,H}
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(out, f!, y, x, cfg)
    else
        chunk_mode_jacobian!(out, f!, y, x, cfg)
    end
    return out
end

@inline jacobian(f::F, x::SArray) where {F} = vector_mode_jacobian(f, x)

@inline jacobian!(out, f::F, x::SArray) where {F} = vector_mode_jacobian!(out, f, x)

#####################
# result extraction #
#####################

@generated function extract_jacobian(ydual::SArray{SY,VY,DY,M},
                                     x::SArray{SX,VX,DX,N}) where {SY,VY,DY,M,SX,VX,DX,N}
    result = Expr(:tuple, [:(partials(ydual[$i], $j)) for i in 1:M, j in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        return SArray{Tuple{M,N}}($result)
    end
end

@generated function extract_value(ydual::SArray{SY,VY,DY,M},
                                     x::SArray{SX,VX,DX,N}) where {SY,VY,DY,M,SX,VX,DX,N}
    result = Expr(:tuple, [:(value(ydual[$i])) for i in 1:M]...)
    return quote
        $(Expr(:meta, :inline))
        return SArray{SX}($result)
    end
end

function extract_jacobian(ydual::AbstractArray, x::SArray{S,V,D,N}) where {S,V,D,N}
    out = similar(ydual, valtype(eltype(ydual)), length(ydual), N)
    return extract_jacobian!(out, ydual, N)
end

function extract_jacobian!(out::AbstractArray, ydual::AbstractArray, n)
    out_reshaped = reshape(out, length(ydual), n)
    for col in 1:size(out_reshaped, 2), row in 1:size(out_reshaped, 1)
        out_reshaped[row, col] = partials(ydual[row], col)
    end
    return out 
end

function extract_jacobian!(out::DiffResult, ydual::AbstractArray, n)
    jout = extract_jacobian!(DiffBase.jacobian(out), ydual, n)
    out = DiffBase.jacobian!(out, jout)
    return out
end

function extract_jacobian_chunk!(out, ydual, index, chunksize)
    offset = index - 1
    for i in 1:chunksize
        col = i + offset
        for row in eachindex(ydual)
            out[row, col] = partials(ydual[row], i)
        end
    end
    return out
end

reshape_jacobian(out, ydual, xdual) = reshape(out, length(ydual), length(xdual))
reshape_jacobian(out::DiffResult, ydual, xdual) = reshape_jacobian(DiffBase.jacobian(out), ydual, xdual)

###############
# vector mode #
###############

function vector_mode_jacobian(f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    ydual = vector_mode_dual_eval(f, x, cfg)
    out = similar(ydual, valtype(eltype(ydual)), length(ydual), N)
    extract_jacobian!(out, ydual, N)
    extract_value!(out, ydual)
    return out
end

function vector_mode_jacobian(f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    ydual = vector_mode_dual_eval(f!, y, x, cfg)
    map!(value, y, ydual)
    out = similar(y, length(y), N)
    extract_jacobian!(out, ydual, N)
    map!(value, y, ydual)
    return out
end

function vector_mode_jacobian!(out, f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    ydual = vector_mode_dual_eval(f, x, cfg)
    extract_jacobian!(out, ydual, N)
    extract_value!(out, ydual)
    return out
end

function vector_mode_jacobian!(out, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    ydual = vector_mode_dual_eval(f!, y, x, cfg)
    map!(value, y, ydual)
    extract_jacobian!(out, ydual, N)
    extract_value!(out, y, ydual)
    return out
end

@inline function vector_mode_jacobian(f::F, x::SArray) where F
    return extract_jacobian(vector_mode_dual_eval(f, x), x)
end

@inline function vector_mode_jacobian!(out, f::F, x::SArray{S,V,D,N}) where {F,S,V,D,N}
    ydual = vector_mode_dual_eval(f, x)
    out = extract_jacobian!(out, ydual, N)
    out = extract_value!(out, ydual)
    return out
end

@inline function vector_mode_jacobian!(out::ImmutableDiffResult, f::F, x::SArray{S,V,D,N}) where {F,S,V,D,N}
    ydual = vector_mode_dual_eval(f, x)
    jout = extract_jacobian(ydual, x)
    vout = extract_value(ydual, DiffBase.value(out))
    out = DiffBase.jacobian!(out, jout)
    out = DiffBase.value!(out, vout)
    return out
end

# chunk mode #
#------------#

function jacobian_chunk_mode_expr(work_array_definition::Expr, compute_ydual::Expr,
                                  out_definition::Expr, y_definition::Expr)
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
        $(out_definition)
        out_reshaped = reshape_jacobian(out, ydual, xdual)
        extract_jacobian_chunk!(out_reshaped, ydual, 1, N)
        seed!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            $(compute_ydual)
            extract_jacobian_chunk!(out_reshaped, ydual, i, N)
            seed!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        $(compute_ydual)
        extract_jacobian_chunk!(out_reshaped, ydual, lastchunkindex, lastchunksize)

        $(y_definition)

        return out
    end
end

@eval function chunk_mode_jacobian(f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(out = similar(ydual, valtype(eltype(ydual)), length(ydual), xlen)),
                               :()))
end

@eval function chunk_mode_jacobian(f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(out = similar(y, length(y), xlen)),
                               :(map!(value, y, ydual))))
end

@eval function chunk_mode_jacobian!(out, f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(),
                               :(extract_value!(out, ydual))))
end

@eval function chunk_mode_jacobian!(out, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(),
                               :(extract_value!(out, y, ydual))))
end
