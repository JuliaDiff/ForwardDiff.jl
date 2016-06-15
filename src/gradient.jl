##################
# GradientResult #
##################

type GradientResult{V,G} <: ForwardDiffResult
    value::V
    gradient::G
end

GradientResult(x) = GradientResult(first(x), similar(x))

value(result::GradientResult) = result.value
gradient(result::GradientResult) = result.gradient

###############
# API methods #
###############

function gradient{N}(f, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        return vector_mode_gradient(f, x, chunk)
    elseif multithread
        return multithread_chunk_mode_gradient(f, x, chunk)
    else
        return chunk_mode_gradient(f, x, chunk)
    end
end

function gradient!{N}(out, f, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        vector_mode_gradient!(out, f, x, chunk)
    elseif multithread
        multithread_chunk_mode_gradient!(out, f, x, chunk)
    else
        chunk_mode_gradient!(out, f, x, chunk)
    end
    return out
end

#######################
# workhorse functions #
#######################

# result extraction #
#-------------------#

@inline load_gradient_value!(out, dual) = out

function load_gradient_value!(out::GradientResult, dual)
    out.value = value(dual)
    return out
end

function load_gradient!(out, dual)
    for i in eachindex(out)
        out[i] = partials(dual, i)
    end
    return out
end

@inline function load_gradient!(out::GradientResult, dual)
    load_gradient!(out.gradient, dual)
    return out
end

function load_gradient_chunk!(out, dual, index, chunksize)
    offset = index - 1
    for i in 1:chunksize
        out[i + offset] = partials(dual, i)
    end
    return out
end

@inline function load_gradient_chunk!(out::GradientResult, dual, index, chunksize)
    load_gradient_chunk!(out.gradient, dual, index, chunksize)
    return out
end

# vector mode #
#-------------#

function compute_vector_mode_gradient(f, x, chunk)
    cache = jacobian_cachefetch!(x, chunk)
    xdual = cache.dualvec
    seed!(xdual, x, cache.seeds, 1)
    return f(xdual)
end

function vector_mode_gradient(f, x, chunk)
    dual = compute_vector_mode_gradient(f, x, chunk)
    out = similar(x, numtype(dual))
    return load_gradient!(out, dual)
end

function vector_mode_gradient!(out, f, x, chunk)
    dual = compute_vector_mode_gradient(f, x, chunk)
    load_gradient_value!(out, dual)
    load_gradient!(out, dual)
    return out
end

# chunk mode #
#------------#

function chunk_mode_gradient_expr(out_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # fetch and seed work vectors
        cache = jacobian_cachefetch!(x, chunk)
        xdual = cache.dualvec
        seeds = cache.seeds
        zeroseed = zero(eltype(seeds))
        seedall!(xdual, x, zeroseed)

        # do first chunk manually to calculate output type
        seed!(xdual, x, seeds, 1)
        dual = f(xdual)
        seed!(xdual, x, zeroseed, 1)
        $(out_definition)
        load_gradient_chunk!(out, dual, 1, N)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, seeds, i)
            dual = f(xdual)
            seed!(xdual, x, zeroseed, i)
            load_gradient_chunk!(out, dual, i, N)
        end

        # do final chunk
        seed!(xdual, x, cache.remainder_seeds, lastchunkindex, lastchunksize)
        dual = f(xdual)
        load_gradient_chunk!(out, dual, lastchunkindex, lastchunksize)

        # load value, this is a no-op unless out is a GradientResult
        load_gradient_value!(out, dual)

        return out
    end
end

@eval function chunk_mode_gradient{N}(f, x, chunk::Chunk{N})
    $(chunk_mode_gradient_expr(:(out = similar(x, numtype(dual)))))
end

@eval function chunk_mode_gradient!{N}(out, f, x, chunk::Chunk{N})
    $(chunk_mode_gradient_expr(:()))
end

# multithreaded chunk mode #
#--------------------------#

if IS_MULTITHREADED_JULIA
    function multithread_chunk_mode_expr(out_definition::Expr)
        return quote
            @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

            # precalculate loop bounds
            xlen = length(x)
            remainder = xlen % N
            lastchunksize = ifelse(remainder == 0, N, remainder)
            lastchunkindex = xlen - lastchunksize + 1
            middlechunks = 2:div(xlen - lastchunksize, N)

            # fetch and seed work vectors
            caches = multithread_jacobian_cachefetch!(x, chunk)
            zeroseed = zero(eltype(seeds))

            Base.Threads.@threads for t in 1:NTHREADS
                seedall!(caches[t].dualvec, x, zeroseed)
            end

            # do first chunk manually to calculate output type
            current_cache = caches[compat_threadid()]
            current_xdual = current_cache.dualvec
            current_seeds = current_cache.seeds
            seed!(current_xdual, x, current_seeds, 1)
            current_dual = f(current_xdual)
            seed!(current_xdual, x, zeroseed, 1)
            $(out_definition)
            load_gradient_chunk!(out, current_dual, 1, N)

            # do middle chunks
            Base.Threads.@threads for c in middlechunks
                # see https://github.com/JuliaLang/julia/issues/14948
                local chunk_cache = caches[compat_threadid()]
                local chunk_xdual = chunk_cache.dualvec
                local chunk_seeds = chunk_cache.seeds
                local chunk_index = ((c - 1) * N + 1)
                seed!(chunk_xdual, x, seeds, chunk_index)
                local chunk_dual = f(chunk_xdual)
                seed!(chunk_xdual, x, zeroseed, chunk_index)
                load_gradient_chunk!(out, chunk_dual, chunk_index, N)
            end

            # do final chunk
            seed!(current_xdual, x, current_cache.remainder_seeds, lastchunkindex, lastchunksize)
            current_dual = f(current_xdual)
            load_gradient_chunk!(out, current_dual, lastchunkindex, lastchunksize)

            # load value, this is a no-op unless `out` is a GradientResult
            load_gradient_value!(out, current_dual)

            return out
        end
    end

    @eval function multithread_chunk_mode_gradient{N}(f, x, chunk::Chunk{N})
        $(multithread_chunk_mode_expr(:(out = similar(x, numtype(dual)))))
    end

    @eval function multithread_chunk_mode_gradient!{N}(out, f, x, chunk::Chunk{N})
        $(multithread_chunk_mode_expr(:()))
    end
else
    function multithread_chunk_mode_gradient(args...)
        error("Multithreading is not enabled for this Julia installation.")
    end
    multithread_chunk_mode_gradient!(args...) = multithread_chunk_mode_gradient()
end
