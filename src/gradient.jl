###############
# API methods #
###############

function gradient(f, x, opts::AbstractOptions = Options(x))
    if chunksize(opts) == length(x)
        return vector_mode_gradient(f, x, opts)
    else
        return chunk_mode_gradient(f, x, opts)
    end
end

function gradient!(out, f, x, opts::AbstractOptions = Options(x))
    if chunksize(opts) == length(x)
        vector_mode_gradient!(out, f, x, opts)
    else
        chunk_mode_gradient!(out, f, x, opts)
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

function vector_mode_gradient(f, x, opts)
    ydual = vector_mode_dual_eval(f, x, opts)
    out = similar(x, valtype(ydual))
    return extract_gradient!(out, ydual)
end

function vector_mode_gradient!(out, f, x, opts)
    ydual = vector_mode_dual_eval(f, x, opts)
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
        xdual = opts.duals
        seeds = opts.seeds
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

@eval function chunk_mode_gradient{N}(f, x, opts::Options{N})
    $(chunk_mode_gradient_expr(:(out = similar(x, valtype(ydual)))))
end

@eval function chunk_mode_gradient!{N}(out, f, x, opts::Options{N})
    $(chunk_mode_gradient_expr(:()))
end

# multithreaded #
#---------------#

if IS_MULTITHREADED_JULIA
    function multithread_chunk_mode_expr(out_definition::Expr)
        return quote
            opts = gradient_options(multi_opts)
            N = chunksize(opts)
            @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

            # precalculate loop bounds
            xlen = length(x)
            remainder = xlen % N
            lastchunksize = ifelse(remainder == 0, N, remainder)
            lastchunkindex = xlen - lastchunksize + 1
            middlechunks = 2:div(xlen - lastchunksize, N)

            # fetch and seed work vectors
            current_opts = opts[compat_threadid()]
            current_xdual = current_opts.duals
            current_seeds = current_opts.seeds

            Base.Threads.@threads for t in 1:length(opts)
                seed!(opts[t].duals, x)
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
                local chunk_opts = opts[compat_threadid()]
                local chunk_xdual = chunk_opts.duals
                local chunk_seeds = chunk_opts.seeds
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

    @eval function chunk_mode_gradient(f, x, multi_opts::Multithread)
        $(multithread_chunk_mode_expr(:(out = similar(x, valtype(current_ydual)))))
    end

    @eval function chunk_mode_gradient!(out, f, x, multi_opts::Multithread)
        $(multithread_chunk_mode_expr(:()))
    end
else
    chunk_mode_gradient(f, x, opts::Tuple) = error("Multithreading is not enabled for this Julia installation.")
    chunk_mode_gradient!(out, f, x, opts::Tuple) = chunk_mode_gradient!(f, x, opts)
end
