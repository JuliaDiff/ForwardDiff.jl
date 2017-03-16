###############
# API methods #
###############

function jacobian{F}(f::F, x, cfg::JacobianConfig = JacobianConfig(x))
    if chunksize(cfg) == length(x)
        return vector_mode_jacobian(f, x, cfg)
    else
        return chunk_mode_jacobian(f, x, cfg)
    end
end

function jacobian{F}(f!::F, y, x, cfg::JacobianConfig = JacobianConfig(y, x))
    if chunksize(cfg) == length(x)
        return vector_mode_jacobian(f!, y, x, cfg)
    else
        return chunk_mode_jacobian(f!, y, x, cfg)
    end
end

function jacobian!{F}(out, f::F, x, cfg::JacobianConfig = JacobianConfig(x))
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(out, f, x, cfg)
    else
        chunk_mode_jacobian!(out, f, x, cfg)
    end
    return out
end

function jacobian!{F}(out, f!::F, y, x, cfg::JacobianConfig = JacobianConfig(y, x))
    if chunksize(cfg) == length(x)
        vector_mode_jacobian!(out, f!, y, x, cfg)
    else
        chunk_mode_jacobian!(out, f!, y, x, cfg)
    end
    return out
end

#####################
# result extraction #
#####################

function extract_jacobian!(out::AbstractArray, ydual::AbstractArray, n)
    out_reshaped = reshape(out, length(ydual), n)
    for col in 1:size(out_reshaped, 2), row in 1:size(out_reshaped, 1)
        out_reshaped[row, col] = partials(ydual[row], col)
    end
    return out
end

function extract_jacobian!(out::DiffResult, ydual::AbstractArray, n)
    extract_jacobian!(DiffBase.jacobian(out), ydual, n)
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

function vector_mode_jacobian{F,N}(f::F, x, cfg::JacobianConfig{N})
    ydual = vector_mode_dual_eval(f, x, cfg)
    out = similar(ydual, valtype(eltype(ydual)), length(ydual), N)
    extract_jacobian!(out, ydual, N)
    extract_value!(out, ydual)
    return out
end

function vector_mode_jacobian{F,N}(f!::F, y, x, cfg::JacobianConfig{N})
    ydual = vector_mode_dual_eval(f!, y, x, cfg)
    map!(value, y, ydual)
    out = similar(y, length(y), N)
    extract_jacobian!(out, ydual, N)
    map!(value, y, ydual)
    return out
end

function vector_mode_jacobian!{F,N}(out, f::F, x, cfg::JacobianConfig{N})
    ydual = vector_mode_dual_eval(f, x, cfg)
    extract_jacobian!(out, ydual, N)
    extract_value!(out, ydual)
    return out
end

function vector_mode_jacobian!{F,N}(out, f!::F, y, x, cfg::JacobianConfig{N})
    ydual = vector_mode_dual_eval(f!, y, x, cfg)
    map!(value, y, ydual)
    extract_jacobian!(out, ydual, N)
    extract_value!(out, y, ydual)
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

@eval function chunk_mode_jacobian{F,N}(f::F, x, cfg::JacobianConfig{N})
    $(jacobian_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(out = similar(ydual, valtype(eltype(ydual)), length(ydual), xlen)),
                               :()))
end

@eval function chunk_mode_jacobian{F,N}(f!::F, y, x, cfg::JacobianConfig{N})
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(out = similar(y, length(y), xlen)),
                               :(map!(value, y, ydual))))
end

@eval function chunk_mode_jacobian!{F,N}(out, f::F, x, cfg::JacobianConfig{N})
    $(jacobian_chunk_mode_expr(quote
                                   xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(),
                               :(extract_value!(out, ydual))))
end

@eval function chunk_mode_jacobian!{F,N}(out, f!::F, y, x, cfg::JacobianConfig{N})
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = cfg.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(),
                               :(extract_value!(out, y, ydual))))
end
