##################
# JacobianResult #
##################

type JacobianResult{V,J} <: ForwardDiffResult
    value::V
    jacobian::J
end

JacobianResult(x) = JacobianResult(similar(x), similar(x, length(x), length(x)))

value(result::JacobianResult) = result.value
jacobian(result::JacobianResult) = result.jacobian

###############
# API methods #
###############

function jacobian{N}(f, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        return vector_mode_jacobian(f, x, chunk)
    elseif multithread
        return multithread_chunk_mode_jacobian(f, x, chunk)
    else
        return chunk_mode_jacobian(f, x, chunk)
    end
end

function jacobian{N}(f!, y, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        return vector_mode_jacobian(f!, y, x, chunk)
    elseif multithread
        return multithread_chunk_mode_jacobian(f!, y, x, chunk)
    else
        return chunk_mode_jacobian(f!, y, x, chunk)
    end
end

function jacobian!{N}(out, f, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        vector_mode_jacobian!(out, f, x, chunk)
    elseif multithread
        multithread_chunk_mode_jacobian!(out, f, x, chunk)
    else
        chunk_mode_jacobian!(out, f, x, chunk)
    end
    return out
end

function jacobian!{N}(out, f!, y, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    @assert !(isa(out, JacobianResult)) "use jacobian!(JacobianResult(out, y), f!, x, ...) instead of jacobian!(out, f!, y, x, ...)"
    if N == length(x)
        vector_mode_jacobian!(out, f!, y, x, chunk)
    elseif multithread
        multithread_chunk_mode_jacobian!(out, f!, y, x, chunk)
    else
        chunk_mode_jacobian!(out, f!, y, x, chunk)
    end
    return out
end

function jacobian!{N}(out::JacobianResult, f!, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    jacobian!(out.jacobian, f!, out.value, x, chunk; multithread = multithread)
    return out
end

#######################
# workhorse functions #
#######################

# result extraction #
#-------------------#

function load_jacobian_value!(y, ydual)
    for i in eachindex(y)
        y[i] = value(ydual[i])
    end
    return y
end

function load_jacobian_value!(out::JacobianResult, ydual)
    load_jacobian_value!(out.value, ydual)
    return out
end

function load_jacobian!(out, ydual)
    for col in 1:size(out, 2), row in 1:size(out, 1)
        out[row, col] = partials(ydual[row], col)
    end
    return out
end

@inline function load_jacobian!(out::JacobianResult, ydual)
    load_jacobian!(out.jacobian, ydual)
    return out
end

function load_jacobian_chunk!(out, ydual, index, chunksize)
    offset = index - 1
    for i in 1:chunksize
        col = i + offset
        for row in eachindex(ydual)
            out[row, col] = partials(ydual[row], i)
        end
    end
    return out
end

@inline function load_jacobian_chunk!(out::JacobianResult, ydual, index, chunksize)
    load_jacobian_chunk!(out.jacobian, ydual, index, chunksize)
    return out
end

# vector mode #
#-------------#

function compute_vector_mode_jacobian(f, x, chunk)
    cache = jacobian_cachefetch!(x, chunk)
    xdual = cache.dualvec
    seed!(xdual, x, cache.seeds, 1)
    return f(xdual)
end

function compute_vector_mode_jacobian(f!, y, x, chunk)
    cache = jacobian_cachefetch!(x, chunk)
    ycache = jacobian_cachefetch!(y, chunk, true)
    xdual = cache.dualvec
    ydual = ycache.dualvec
    seed!(xdual, x, cache.seeds, 1)
    seedall!(ydual, y, zero(eltype(ycache.seeds)))
    f!(ydual, xdual)
    return ydual
end

function vector_mode_jacobian{N}(f, x, chunk::Chunk{N})
    ydual = compute_vector_mode_jacobian(f, x, chunk)
    out = similar(ydual, numtype(eltype(ydual)), length(ydual), N)
    return load_jacobian!(out, ydual)
end

function vector_mode_jacobian{N}(f!, y, x, chunk::Chunk{N})
    ydual = compute_vector_mode_jacobian(f!, y, x, chunk)
    load_jacobian_value!(y, ydual)
    out = similar(y, length(y), N)
    return load_jacobian!(out, ydual)
end

function vector_mode_jacobian!(out, f, x, chunk)
    ydual = compute_vector_mode_jacobian(f, x, chunk)
    return load_jacobian!(out, ydual)
end

function vector_mode_jacobian!(out, f!, y, x, chunk)
    ydual = compute_vector_mode_jacobian(f!, y, x, chunk)
    load_jacobian_value!(y, ydual)
    return load_jacobian!(out, ydual)
end

# chunk mode #
#------------#

function jacobian_chunk_mode_expr(out_definition::Expr, cache_definition::Expr,
                                  ydual_compute::Expr, y_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # fetch and seed work vectors
        $(cache_definition)
        xdual = cache.dualvec
        seeds = cache.seeds
        zeroseed = zero(eltype(seeds))
        seedall!(xdual, x, zeroseed)

        # do first chunk manually to calculate output type
        seed!(xdual, x, seeds, 1)
        $(ydual_compute)
        seed!(xdual, x, zeroseed, 1)
        $(out_definition)
        load_jacobian_chunk!(out, ydual, 1, N)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, seeds, i)
            $(ydual_compute)
            seed!(xdual, x, zeroseed, i)
            load_jacobian_chunk!(out, ydual, i, N)
        end

        # do final chunk
        seed!(xdual, x, cache.remainder_seeds, lastchunkindex, lastchunksize)
        $(ydual_compute)
        load_jacobian_chunk!(out, ydual, lastchunkindex, lastchunksize)

        $(y_definition)

        return out
    end
end

@eval function chunk_mode_jacobian{N}(f, x, chunk::Chunk{N})
    $(jacobian_chunk_mode_expr(:(out = similar(x, numtype(eltype(ydual)), length(ydual), xlen)),
                               :(cache = jacobian_cachefetch!(x, chunk)),
                               :(ydual = f(xdual)),
                               :()))
end

@eval function chunk_mode_jacobian{N}(f!, y, x, chunk::Chunk{N})
    $(jacobian_chunk_mode_expr(:(out = similar(y, numtype(eltype(ydual)), length(ydual), xlen)),
                               quote
                                   cache = jacobian_cachefetch!(x, chunk)
                                   ycache = jacobian_cachefetch!(y, chunk, true)
                                   ydual = ycache.dualvec
                                   yzeroseed = zero(eltype(ycache.seeds))
                               end,
                               :(f!(seedall!(ydual, y, yzeroseed), xdual)),
                               :(load_jacobian_value!(y, ydual))))
end

@eval function chunk_mode_jacobian!{N}(out, f, x, chunk::Chunk{N})
    $(jacobian_chunk_mode_expr(:(),
                               :(cache = jacobian_cachefetch!(x, chunk)),
                               :(ydual = f(xdual)),
                               :()))
end

@eval function chunk_mode_jacobian!{N}(out, f!, y, x, chunk::Chunk{N})
    $(jacobian_chunk_mode_expr(:(),
                               quote
                                   cache = jacobian_cachefetch!(x, chunk)
                                   ycache = jacobian_cachefetch!(y, chunk, true)
                                   ydual = ycache.dualvec
                                   yzeroseed = zero(eltype(ycache.seeds))
                               end,
                               :(f!(seedall!(ydual, y, yzeroseed), xdual)),
                               :(load_jacobian_value!(y, ydual))))
end
