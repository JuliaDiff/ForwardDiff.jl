#################
# HessianResult #
#################

type HessianResult{V,G,H} <: ForwardDiffResult
    value::V
    gradient::G
    hessian::H
end

value(result::HessianResult) = result.value

gradient(result::HessianResult) = result.gradient

gradient!(out, result::HessianResult) = copy!(out, result.gradient)

hessian(result::HessianResult) = result.hessian

hessian!(out, result::HessianResult) = copy!(out, result.hessian)

###############
# API methods #
###############

function hessian{N}(f, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        return vector_mode_hessian(f, x, chunk)
    elseif multithread
        return multithread_chunk_mode_hessian(f, x, chunk)
    else
        return chunk_mode_hessian(f, x, chunk)
    end
end

function hessian!{N}(out, f, x, chunk::Chunk{N} = pickchunk(x); multithread = false)
    if N == length(x)
        vector_mode_hessian!(out, f, x, chunk)
    elseif multithread
        multithread_chunk_mode_hessian!(out, f, x, chunk)
    else
        chunk_mode_hessian!(out, f, x, chunk)
    end
    return out
end

#######################
# workhorse functions #
#######################

# result extraction #
#-------------------#

@inline load_hessian_value!(out, dual) = out

function load_hessian_value!(out::HessianResult, dual)
    out.value = value(value(dual))
    return out
end

@inline load_hessian_gradient!(out, dual) = out

function load_hessian_gradient!(out::HessianResult, dual)
    grad = out.gradient
    val = value(dual)
    for i in eachindex(grad)
        grad[i] = partials(val, i)
    end
    return out
end

@inline load_hessian_gradient_chunk!(out, dual, index, chunksize) = out

function load_hessian_gradient_chunk!(out::HessianResult, dual, index, chunksize)
    load_gradient_chunk!(out.gradient, value(dual), index, chunksize)
    return out
end

function load_hessian!(out, dual)
    for col in 1:size(out, 2), row in 1:size(out, 1)
        out[row, col] = partials(dual, row, col)
    end
    return out
end

@inline function load_hessian!(out::HessianResult, dual)
    load_hessian!(out.hessian, dual)
    return out
end

function load_hessian_diagonal_chunk!(out, dual, index, chunksize)
    offset = index - 1
    for j in 1:chunksize, i in 1:chunksize
        out[i+offset, j+offset] = partials(dual, i, j)
    end
    return out
end

@inline function load_hessian_diagonal_chunk!(out::HessianResult, dual, index, chunksize)
    load_hessian_diagonal_chunk!(out.hessian, dual, index, chunksize)
    return out
end

function load_hessian_side_chunk!(out, dual, i, j, chunksize)
    offset = i - 1
    extrachunk = chunksize + 1
    for k in 1:chunksize
        out[k+offset, j] = partials(dual, k, extrachunk)
        out[j, k+offset] = partials(dual, extrachunk, k)
    end
    return out
end

@inline function load_hessian_side_chunk!(out::HessianResult, dual, i, j, chunksize)
    load_hessian_side_chunk!(out.hessian, dual, i, j, chunksize)
    return out
end

# vector mode #
#-------------#

function compute_vector_mode_hessian(f, x, chunk)
    xdual = fetchdualvechess(x, chunk)
    inseeds = fetchseeds(numtype(eltype(xdual)))
    outseeds = fetchseeds(eltype(xdual))
    seed!(xdual, x, inseeds, outseeds, 1)
    return f(xdual)
end

function vector_mode_hessian(f, x, chunk)
    dual = compute_vector_mode_hessian(f, x, chunk)
    out = similar(x, numtype(numtype(dual)), length(x), length(x))
    return load_hessian!(out, dual)
end

function vector_mode_hessian!(out, f, x, chunk)
    dual = compute_vector_mode_hessian(f, x, chunk)
    load_hessian_value!(out, dual)
    load_hessian_gradient!(out, dual)
    load_hessian!(out, dual)
    return out
end

# chunk mode #
#------------#

function chunk_mode_hessian_expr(out_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # diagonal chunks #
        #-----------------#
        # We first compute the derivatives in blocks along
        # the diagonal. The size of these blocks is
        # determined by the chunk size.
        #
        # For example, if `chunk = 2` and `length(x) = 7`,
        # the numbers inside the slots below indicate the
        # iteration (i.e. `i`th call to `f`) in which they
        # are filled:
        #
        # 7x7 Hessian with chunk=2:
        # -----------------------------
        # | 1 | 1 |   |   |   |   |   |
        # -----------------------------
        # | 1 | 1 |   |   |   |   |   |
        # -----------------------------
        # |   |   | 2 | 2 |   |   |   |
        # -----------------------------
        # |   |   | 2 | 2 |   |   |   |
        # -----------------------------
        # |   |   |   |   | 3 | 3 |   |
        # -----------------------------
        # |   |   |   |   | 3 | 3 |   |
        # -----------------------------
        # |   |   |   |   |   |   | 4 |
        # -----------------------------

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        nfullchunks = div(xlen - lastchunksize, N)

        # fetch and seed work vectors
        xdual = fetchdualvechess(x, chunk)
        inseeds = fetchseeds(Dual{N,T})
        outseeds = fetchseeds(Dual{N,Dual{N,T}})
        inzero = zero(Partials{N,T})
        outzero = zero(Partials{N,Dual{N,T}})
        seedall!(xdual, x, inzero, outzero)

        # do first chunk manually for dynamic output definition
        seed!(xdual, x, inseeds, outseeds, 1)
        dual = f(xdual)
        $(out_definition)
        load_hessian_diagonal_chunk!(out, dual, 1, N)
        load_hessian_gradient_chunk!(out, dual, 1, N)
        seed!(xdual, x, inzero, outzero, 1)

        # do middle chunks
        for c in 2:nfullchunks
            i = (c - 1) * N + 1
            seed!(xdual, x, inseeds, outseeds, i)
            dual = f(xdual)
            seed!(xdual, x, inzero, outzero, i)
            load_hessian_diagonal_chunk!(out, dual, i, N)
            load_hessian_gradient_chunk!(out, dual, i, N)
        end

        # do final chunk
        if lastchunksize != N
            seeds = fetchseeds(eltype(xdual), Chunk{lastchunksize}())
        end
        seed!(xdual, x, inseeds, outseeds, lastchunkindex, lastchunksize)
        dual = f(xdual)
        load_hessian_diagonal_chunk!(out, dual, lastchunkindex, lastchunksize)
        load_hessian_gradient_chunk!(out, dual, lastchunkindex, lastchunksize)
        load_hessian_value!(out, dual)

        # off-diagonal chunks #
        #---------------------#
        # Now, we fill in the off-diagonal chunks. Like
        # the previous diagram, the numbers inside the
        # slots indicate the iteration (i.e. `i`th call
        # to `f`) in which they are filled:
        #
        # 7x7 Hessian with chunk=2:
        # -----------------------------
        # |   |   | 1 | 2 | 3 | 4 | 5 |
        # -----------------------------
        # |   |   | 1 | 2 | 3 | 4 | 5 |
        # -----------------------------
        # | 1 | 1 |   |   | 6 | 7 | 8 |
        # -----------------------------
        # | 2 | 2 |   |   | 6 | 7 | 8 |
        # -----------------------------
        # | 3 | 3 | 6 | 6 |   |   | 9 |
        # -----------------------------
        # | 4 | 4 | 7 | 7 |   |   | 9 |
        # -----------------------------
        # | 5 | 5 | 8 | 8 | 9 | 9 |   |
        # -----------------------------

        sideN = N + 1

        xdual = fetchdualvechess(x, Chunk{sideN}())
        inseeds = fetchseeds(Dual{sideN,T})
        outseeds = fetchseeds(Dual{sideN,Dual{sideN,T}})
        lastinseed = last(inseeds)
        lastoutseed = last(outseeds)
        inzero = zero(Partials{sideN,T})
        outzero = zero(Partials{sideN,Dual{sideN,T}})
        seedall!(xdual, x, inzero, outzero)

        for c in 1:(nfullchunks + 1)
            i = (c - 1) * N + 1
            sideseed!(xdual, x, inseeds, outseeds, i)
            for j in (i + N):(xlen)
                sideseedj!(xdual, x, lastinseed, lastoutseed, j)
                dual = f(xdual)
                load_hessian_side_chunk!(out, dual, i, j, N)
                sideseedj!(xdual, x, inzero, outzero, j)
            end
            sideseed!(xdual, x, inzero, outzero, i)
        end

        return out
    end
end

@eval function chunk_mode_hessian{N}(f, x, chunk::Chunk{N})
    $(chunk_mode_hessian_expr(:(out = similar(x, numtype(numtype(dual)), xlen, xlen))))
end

@eval function chunk_mode_hessian!{N}(out, f, x, chunk::Chunk{N})
    $(chunk_mode_hessian_expr(:()))
end
