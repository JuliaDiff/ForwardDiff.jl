###################
# Taking Hessians #
###################

# Exposed API methods #
#---------------------#
function hessian!{T}(output::Matrix{T},
                     f,
                     x::Vector;
                     chunk_size::Int=default_chunk,
                     cache::ForwardDiffCache=dummy_cache)
    xlen = length(x)
    @assert (xlen, xlen) == size(output) "The output matrix must have size (length(input), length(input))"
    return _calc_hessian!(output, f, x, Val{xlen}, Val{chunk_size}, cache)::Matrix{T}
end

function hessian{T}(f,
                    x::Vector{T};
                    chunk_size::Int=default_chunk,
                    cache::ForwardDiffCache=dummy_cache)
    xlen = length(x)
    output = similar(x, xlen, xlen)
    return _calc_hessian!(output, f, x, Val{xlen}, Val{chunk_size}, cache)::Matrix{T}
end

function hessian(f; mutates=false)
    cache = ForwardDiffCache()
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
gradnum_type{N,T,C}(::Type{HessianNumber{N,T,C}}) = GradientNumber{N,T,C}

@generated function _calc_hessian!{S,T,xlen,chunk_size}(output::Matrix{S}, f, x::Vector{T}, 
                                                        X::Type{Val{xlen}}, C::Type{Val{chunk_size}},
                                                        cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)
    full_bool = chunk_size_matches_full(xlen, chunk_size)
    
    # chunk_size is incremented by one when users 
    # input a non-xlen chunk_size (this allows 
    # simplification of loop alignment for the 
    # chunk-based calculation code)
    C2 = Val{full_bool ? chunk_size : chunk_size+1}
    H = workvec_eltype(HessianNumber, T, Val{xlen}, C2)
    G = gradnum_type(H)

    if full_bool
        body = quote
            @simd for i in eachindex(x)
                @inbounds hessvec[i] = HessianNumber(G(x[i], partials[i]), hesszeros)
            end

            result::ResultType = f(hessvec)

            q = 1
            for i in 1:N
                for j in 1:i
                    val = hess(result, q)
                    @inbounds output[i, j] = val
                    @inbounds output[j, i] = val
                    q += 1
                end
            end
        end
    else
        body = quote
            M = N-1
            gradzeros = get_zeros!(cache, G)

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

                chunk_result::ResultType = f(hessvec)
                
                q = 1
                for j in i:(i+M-1)
                    for k in i:j
                        val = hess(chunk_result, q)
                        @inbounds output[j, k] = val
                        @inbounds output[k, j] = val
                        q += 1
                    end
                end

                offset = i-1
                @simd for j in 1:M
                    q = j+offset
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

                        chunk_result::ResultType = f(hessvec)

                        for i in 1:M
                            row = row_offset + i
                            q = halfhesslen(i) + 1
                            val = hess(chunk_result, q)
                            @inbounds output[row, col] = val
                            @inbounds output[col, row] = val
                            @inbounds hessvec[row] = HessianNumber(G(x[row], gradzeros), hesszeros)
                        end
                    end
                    @inbounds hessvec[col] = HessianNumber(G(x[col], gradzeros), hesszeros)
                end
            end
        end
    end

    return quote 
        H = $H
        G = $G
        N = npartials(H)
        ResultType = switch_eltype(H, S)

        hessvec = get_workvec!(cache, HessianNumber, T, X, $C2)
        partials = get_partials!(cache, H)
        hesszeros = get_zeros!(cache, H)
        
        $body

        return output::Matrix{S}
    end
end