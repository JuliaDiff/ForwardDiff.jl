###############
# API methods #
###############

function gradient{F}(f::F, x, cfg::AbstractConfig = Config(x))
    if chunksize(cfg) == length(x)
        return vector_mode_gradient(f, x, cfg)
    else
        return chunk_mode_gradient(f, x, cfg)
    end
end

function gradient!{F}(out, f::F, x, cfg::AbstractConfig = Config(x))
    if chunksize(cfg) == length(x)
        vector_mode_gradient!(out, f, x, cfg)
    else
        chunk_mode_gradient!(out, f, x, cfg)
    end
    return out
end

#####################
# result extraction #
#####################

function extract_gradient!(out::DiffResult, y::Real)
    DiffBase.value!(out, y)
    grad = DiffBase.gradient(out)
    fill!(grad, zero(y))
    return out
end

function extract_gradient!(out::DiffResult, dual::Dual)
    DiffBase.value!(out, value(dual))
    DiffBase.gradient!(out, partials(dual))
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

function vector_mode_gradient{F}(f::F, x, cfg)
    ydual = vector_mode_dual_eval(f, x, cfg)
    out = similar(x, valtype(ydual))
    return extract_gradient!(out, ydual)
end

function vector_mode_gradient!{F}(out, f::F, x, cfg)
    ydual = vector_mode_dual_eval(f, x, cfg)
    extract_gradient!(out, ydual)
    return out
end

##############
# chunk mode #
##############

# single threaded #
#-----------------#

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
        seed!(xdual, x, 1)
        $(out_definition)
        extract_gradient_chunk!(out, ydual, 1, N)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            ydual = f(xdual)
            seed!(xdual, x, i)
            extract_gradient_chunk!(out, ydual, i, N)
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

@eval function chunk_mode_gradient{F,N}(f::F, x, cfg::Config{N})
    $(chunk_mode_gradient_expr(:(out = similar(x, valtype(ydual)))))
end

@eval function chunk_mode_gradient!{F,N}(out, f::F, x, cfg::Config{N})
    $(chunk_mode_gradient_expr(:()))
end

# multithreaded #
#---------------#

if IS_MULTITHREADED_JULIA
    function multithread_chunk_mode_expr(out_definition::Expr)
        return quote
            cfg = gradient_options(multi_cfg)
            N = chunksize(cfg)
            @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

            # precalculate loop bounds
            xlen = length(x)
            remainder = xlen % N
            lastchunksize = ifelse(remainder == 0, N, remainder)
            lastchunkindex = xlen - lastchunksize + 1
            middlechunks = 2:div(xlen - lastchunksize, N)

            # fetch and seed work vectors
            current_cfg = cfg[compat_threadid()]
            current_xdual = current_cfg.duals
            current_seeds = current_cfg.seeds

            Base.Threads.@threads for t in 1:length(cfg)
                seed!(cfg[t].duals, x)
            end

            # do first chunk manually to calculate output type
            seed!(current_xdual, x, 1, current_seeds)
            current_ydual = f(current_xdual)
            seed!(current_xdual, x, 1)
            $(out_definition)
            extract_gradient_chunk!(out, current_ydual, 1, N)

            # do middle chunks
            Base.Threads.@threads for c in middlechunks
                # see https://github.com/JuliaLang/julia/issues/14948
                local chunk_cfg = cfg[compat_threadid()]
                local chunk_xdual = chunk_cfg.duals
                local chunk_seeds = chunk_cfg.seeds
                local chunk_index = ((c - 1) * N + 1)
                seed!(chunk_xdual, x, chunk_index, chunk_seeds)
                local chunk_dual = f(chunk_xdual)
                seed!(chunk_xdual, x, chunk_index)
                extract_gradient_chunk!(out, chunk_dual, chunk_index, N)
            end

            # do final chunk
            seed!(current_xdual, x, lastchunkindex, current_seeds, lastchunksize)
            current_ydual = f(current_xdual)
            extract_gradient_chunk!(out, current_ydual, lastchunkindex, lastchunksize)

            # load value, this is a no-op unless `out` is a DiffResult
            extract_value!(out, current_ydual)

            return out
        end
    end

    @eval function chunk_mode_gradient{F}(f::F, x, multi_cfg::Multithread)
        $(multithread_chunk_mode_expr(:(out = similar(x, valtype(current_ydual)))))
    end

    @eval function chunk_mode_gradient!{F}(out, f::F, x, multi_cfg::Multithread)
        $(multithread_chunk_mode_expr(:()))
    end
else
    chunk_mode_gradient(f, x, cfg::Tuple) = error("Multithreading is not enabled for this Julia installation.")
    chunk_mode_gradient!(out, f, x, cfg::Tuple) = chunk_mode_gradient!(f, x, cfg)
end
