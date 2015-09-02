###################
# Taking Hessians #
###################

# Exposed API methods #
#---------------------#
@generated function hessian!{T,A}(output::Matrix{T}, f, x::Vector, ::Type{A}=Void;
                                  chunk_size::Int=default_chunk_size,
                                  cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(hessian!(output, result)::Matrix{T})
    elseif A <: AllResults
        return_stmt = :(hessian!(output, result)::Matrix{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.hessian")
    end

    return quote
        result = _calc_hessian(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

@generated function hessian{T,A}(f, x::Vector{T}, ::Type{A}=Void;
                                 chunk_size::Int=default_chunk_size,
                                 cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(hessian(result)::Matrix{T})
    elseif A <: AllResults
        return_stmt = :(hessian(result)::Matrix{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.hessian")
    end

    return quote
        result = _calc_hessian(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

function hessian{A}(f, ::Type{A}=Void;
                    mutates::Bool=false,
                    chunk_size::Int=default_chunk_size,
                    cache::ForwardDiffCache=ForwardDiffCache())
    if mutates
        function h!(output::Matrix, x::Vector)
            return ForwardDiff.hessian!(output, f, x, A;
                                        chunk_size=chunk_size,
                                        cache=cache)
        end
        return h!
    else
        function h(x::Vector)
            return ForwardDiff.hessian(f, x, A;
                                       chunk_size=chunk_size,
                                       cache=cache)
        end
        return h
    end
end

# Calculate Hessian of a given function #
#---------------------------------------#
function _calc_hessian{S}(f, x::Vector, ::Type{S},
                          chunk_size::Int,
                          cache::ForwardDiffCache)
    X = Val{length(x)}
    C = Val{chunk_size}
    return _calc_hessian(f, x, S, X, C, cache)
end

gradnum_type{N,T,C}(::Type{HessianNumber{N,T,C}}) = GradientNumber{N,T,C}

@generated function _calc_hessian{T,S,xlen,chunk_size}(f, x::Vector{T}, ::Type{S},
                                                       X::Type{Val{xlen}},
                                                       C::Type{Val{chunk_size}},
                                                       cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)

    # chunk_size is incremented by one when users
    # input a non-xlen chunk_size (this allows
    # simplification of loop alignment for the
    # chunk-based calculation code)
    vec_mode_bool = chunk_size_matches_vec_mode(xlen, chunk_size)
    C2 = Val{vec_mode_bool ? chunk_size : chunk_size+1}
    H = workvec_eltype(HessianNumber, T, Val{xlen}, C2)
    N = npartials(H)
    G = gradnum_type(H)

    if vec_mode_bool
        # Vector-Mode
        ResultHess = switch_eltype(H, S)
        body = quote
            @simd for i in eachindex(x)
                @inbounds hessvec[i] = HessianNumber(G(x[i], partials[i]), hesszeros)
            end

            result::$ResultHess = f(hessvec)
        end
    else
        # Chunk-Mode
        ChunkType = switch_eltype(H, S)
        ResultGrad = GradientNumber{xlen,S,Vector{S}}
        ResultHess = HessianNumber{xlen,S,Vector{S}}
        body = quote
            N = $N
            M = N-1
            gradzeros = get_zeros!(cache, G)
            output_grad = Vector{S}(xlen)
            output_hess = Vector{S}(halfhesslen(xlen))

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

            local chunk_result::$ChunkType

            for i in 1:M:xlen
                offset = i-1

                @simd for j in 1:M
                    q = j+offset
                    @inbounds hessvec[q] = HessianNumber(G(x[q], partials[j]), hesszeros)
                end

                chunk_result = f(hessvec)
                
                q = 1
                for j in i:(M+offset)
                    for k in i:j
                        @inbounds output_hess[hess_inds(j, k)] = hess(chunk_result, q)
                        q += 1
                    end
                end

                @simd for j in 1:M
                    q = j+offset
                    @inbounds output_grad[q] = grad(chunk_result, j)
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
                            @inbounds output_hess[hess_inds(row, col)] = hess(chunk_result, q)
                            @inbounds hessvec[row] = HessianNumber(G(x[row], gradzeros), hesszeros)
                        end
                    end
                    @inbounds hessvec[col] = HessianNumber(G(x[col], gradzeros), hesszeros)
                end
            end

            result::$ResultHess = ($ResultHess)(($ResultGrad)(value(chunk_result), output_grad), output_hess)
        end
    end

    return quote
        H, G = $H, $G
        hessvec = get_workvec!(cache, HessianNumber, T, X, $C2)
        partials = get_partials!(cache, H)
        hesszeros = get_zeros!(cache, H)
        
        $body

        return ForwardDiffResult(result)
    end
end
