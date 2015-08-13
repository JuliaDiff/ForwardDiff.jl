###################
# Taking Hessians #
###################
HessianCache() = ForwardDiffCache(HessianNumber)
const hess_void_cache = void_cache(HessianNumber)

# Exposed API methods #
#---------------------#
function hessian!{T}(output::Matrix{T},
                     f,
                     x::Vector;
                     chunk_size::Int=default_chunk,
                     cache::ForwardDiffCache=hess_void_cache)
    xlen = length(x)
    @assert (xlen, xlen) == size(output) "The output matrix must have size (length(input), length(input))"
    return _take_hessian!(output, f, x, chunk_size, cache)::Matrix{T}
end

function hessian{T}(f,
                    x::Vector{T};
                    chunk_size::Int=default_chunk,
                    cache::ForwardDiffCache=hess_void_cache)
    xlen = length(x)
    output = similar(x, xlen, xlen)
    return _take_hessian!(output, f, x, chunk_size, cache)::Matrix{T}
end

function hessian(f; mutates=false)
    cache = HessianCache()
    if mutates
        function hessf!{T}(output::Matrix{T}, x::Vector; chunk_size::Int=default_chunk)
            return hessian!(output, f, x, chunk_size=chunk_size, cache=cache)::Matrix{T}
        end
        return hessf!
    else
        function hessf{T}(x::Vector{T}; chunk_size::Int=default_chunk)
            return hessian(f, x, chunk_size=chunk_size, cache=cache)::Matrix{T}
        end
        return hessf
    end
end

# Calculate Hessian of a given function #
#---------------------------------------#
gradnum_type{N,T,C}(::Vector{HessianNumber{N,T,C}}) = GradientNumber{N,T,C}

function _take_hessian!{T}(output::Matrix{T}, f, x::Vector, chunk_size::Int, cache::ForwardDiffCache)
    full_bool = chunk_size_matches_full(x, chunk_size)
    N = full_bool ? chunk_size : chunk_size+1
    hessvec = get_workvec!(cache, x, N)
    partials = get_partials!(cache, eltype(hessvec))
    hesszeros = get_zeros!(cache, eltype(hessvec))
    if full_bool
        return _calc_hessian_full!(output, f, x, hessvec, partials, hesszeros)::Matrix{T}
    else
        gradzeros = get_zeros!(cache, gradnum_type(hessvec))
        return _calc_hessian_chunks!(output, f, x, hessvec, partials, hesszeros, gradzeros)::Matrix{T}
    end
end

function _calc_hessian_full!{S,N,T,C}(output::Matrix{S},
                                      f,
                                      x::Vector{T},
                                      hessvec::Vector{HessianNumber{N,T,C}},
                                      partials, hesszeros) 
    G = gradnum_type(hessvec)

    @simd for i in eachindex(hessvec)
        @inbounds hessvec[i] = HessianNumber(G(x[i], partials[i]), hesszeros)
    end

    result = f(hessvec)

    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(result, q)::S
            @inbounds output[i, j] = val
            @inbounds output[j, i] = val
            q += 1
        end
    end

    return output::Matrix{S}
end

function _calc_hessian_chunks!{S,N,T,C}(output::Matrix{S},
                                        f,
                                        x::Vector{T},
                                        hessvec::Vector{HessianNumber{N,T,C}},
                                        partials, hesszeros, gradzeros)
    xlen = length(x)
    G = GradientNumber{N,T,C}

    # Keep in mind that chunk_size is internally 
    # incremented by one for HessianNumbers when users 
    # input a non-xlen chunk_size (this allows 
    # simplification of loop alignment for 
    # the chunk evaluating code below)
    M = N-1
    check_chunk_size(x, M)

    @simd for i in eachindex(x)
        @inbounds hessvec[i] = HessianNumber(G(x[i], gradzeros), hesszeros) 
    end

    # The below loop fills triangular blocks 
    # along diagonal. The size of these blocks
    # is determined by the chunk size.
    #
    # For example, if N = 3 and xlen = 6, the 
    # numbers inside the slots below indicate the
    # iteration of the loop (i.e. ith call of f) 
    # in which they are filled:
    #
    # Hessian matrix:
    # -------------------------
    # | 1 | 1 |   |   |   |   |
    # -------------------------
    # | 1 | 1 |   |   |   |   |
    # -------------------------
    # |   |   | 2 | 2 |   |   |
    # -------------------------
    # |   |   | 2 | 2 |   |   |
    # -------------------------
    # |   |   |   |   | 3 | 3 |
    # -------------------------
    # |   |   |   |   | 3 | 3 |
    # -------------------------
    for i in 1:M:xlen
        @simd for j in 1:M
            q = i+j-1
            @inbounds hessvec[q] = HessianNumber(G(x[q], partials[j]), hesszeros)
        end

        chunk_result = f(hessvec)
        
        q = 1
        for j in i:(i+M-1)
            for k in i:j
                val = hess(chunk_result, q)::S
                @inbounds output[j, k] = val
                @inbounds output[k, j] = val
                q += 1
            end
        end

        @simd for j in 1:M
            q = i+j-1
            @inbounds hessvec[q] = HessianNumber(G(x[q], gradzeros), hesszeros)
        end
    end

    # The below loop fills in the rest. Once 
    # again, using N = 3 and xlen = 6, with each 
    # iteration (i.e. ith call of f) filling the 
    # corresponding slots, and where 'x' indicates
    # previously filled slots:
    #
    # -------------------------
    # | x | x | 1 | 1 | 2 | 2 |
    # -------------------------
    # | x | x | 3 | 3 | 4 | 4 |
    # -------------------------
    # | 1 | 3 | x | x | 5 | 5 |
    # -------------------------
    # | 1 | 3 | x | x | 6 | 6 |
    # -------------------------
    # | 2 | 4 | 5 | 6 | x | x |
    # -------------------------
    # | 2 | 4 | 5 | 6 | x | x |
    # -------------------------
    for offset in M:M:(xlen-M)
        col_offset = offset - M
        for j in 1:M
            col = col_offset + j
            @inbounds hessvec[col] = HessianNumber(G(x[col], partials[1]), hesszeros)
            for row_offset in offset:M:(xlen-1)
                for i in 1:M
                    row = row_offset + i
                    @inbounds hessvec[row] = HessianNumber(G(x[row], partials[i+1]), hesszeros)
                end

                chunk_result = f(hessvec)

                for i in 1:M
                    row = row_offset + i
                    q = halfhesslen(i) + 1
                    val = hess(chunk_result, q)::S
                    @inbounds output[row, col] = val
                    @inbounds output[col, row] = val
                    @inbounds hessvec[row] = HessianNumber(G(x[row], gradzeros), hesszeros)
                end
            end
            @inbounds hessvec[col] = HessianNumber(G(x[col], gradzeros), hesszeros)
        end
    end

    return output::Matrix{S}
end
