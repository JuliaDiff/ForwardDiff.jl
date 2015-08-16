####################
# Taking Jacobians #
####################

# Exposed API methods #
#---------------------#
function jacobian!{T}(output::Matrix{T},
                      f,
                      x::Vector;
                      chunk_size::Int=default_chunk,
                      cache::ForwardDiffCache=dummy_cache)
    xlen = length(x)
    @assert xlen == size(output, 2) "The output matrix must have a number of columns equal to the length of the input vector"
    return _calc_jacobian!(output, f, x, Val{xlen}, Val{chunk_size}, cache)::Matrix{T}
end

function jacobian{T}(f,
                     x::Vector{T};
                     chunk_size::Int=default_chunk,
                     cache::ForwardDiffCache=dummy_cache)
    return _calc_jacobian(f, x, Val{length(x)}, Val{chunk_size}, cache)::Matrix{T}
end

function jacobian(f; mutates=false)
    cache = ForwardDiffCache()
    if mutates
        function jacf!{T}(output::Matrix{T}, x::Vector; chunk_size::Int=default_chunk)
            return ForwardDiff.jacobian!(output, f, x, chunk_size=chunk_size, cache=cache)::Matrix{T}
        end
        return jacf!
    else
        function jacf{T}(x::Vector{T}; chunk_size::Int=default_chunk)
            return ForwardDiff.jacobian(f, x, chunk_size=chunk_size, cache=cache)::Matrix{T}
        end
        return jacf
    end
end

# Calculate Jacobian of a given function #
#----------------------------------------#
@generated function _calc_jacobian!{S,T,xlen,chunk_size}(output::Matrix{S}, f, x::Vector{T}, 
                                                         X::Type{Val{xlen}}, C::Type{Val{chunk_size}},
                                                         cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)

    full_bool = chunk_size_matches_full(xlen, chunk_size)
    G = workvec_eltype(GradientNumber, T, Val{xlen}, Val{chunk_size})

    if full_bool
        body = quote
            @simd for i in eachindex(x)
                @inbounds gradvec[i] = G(x[i], partials[i])
            end

            result::Vector{ResultType} = f(gradvec)

            for i in eachindex(result), j in eachindex(x)
                output[i,j] = grad(result[i], j)
            end
        end
    else
        body = quote
            gradzeros = get_zeros!(cache, G)

            @simd for i in eachindex(x)
                @inbounds gradvec[i] = G(x[i], gradzeros)
            end

            for i in 1:N:xlen
                offset = i-1

                @simd for j in 1:N
                    m = j+offset
                    @inbounds gradvec[m] = G(x[m], partials[j])
                end
                
                chunk_result::Vector{ResultType} = f(gradvec)

                for j in 1:N
                    m = j+offset
                    for n in 1:size(output,1)
                        @inbounds output[n,m] = grad(chunk_result[n], j)
                    end
                    @inbounds gradvec[m] = G(x[m], gradzeros)
                end
            end
        end
    end

    return quote
        G = $G
        N = npartials(G)
        ResultType = switch_eltype(G, S)

        gradvec = get_workvec!(cache, GradientNumber, T, X, C)
        partials = get_partials!(cache, G)
        
        $body

        return output::Matrix{S}
    end
end

@generated function _calc_jacobian{T,xlen,chunk_size}(f, x::Vector{T}, 
                                                      X::Type{Val{xlen}}, C::Type{Val{chunk_size}},
                                                      cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)

    full_bool = chunk_size_matches_full(xlen, chunk_size)
    G = workvec_eltype(GradientNumber, T, Val{xlen}, Val{chunk_size})

    if full_bool
        body = quote
            @simd for i in eachindex(x)
                @inbounds gradvec[i] = G(x[i], partials[i])
            end

            result::Vector{ResultType} = f(gradvec)

            output = Array(T, length(result), length(x))

            for i in eachindex(result), j in eachindex(x)
                output[i,j] = grad(result[i], j)
            end
        end
    else
        body = quote
            gradzeros = get_zeros!(cache, G)
            
            @simd for i in eachindex(x)
                @inbounds gradvec[i] = G(x[i], gradzeros)
            end

            # Perform the first chunk "manually" so that
            # we get to inspect the size of the output
            @simd for j in 1:N
                m = j
                @inbounds gradvec[m] = G(x[m], partials[j])
            end
            
            first_result::Vector{ResultType} = f(gradvec)

            # Now that we know the shape of the first result,
            # we can construct an output matrix
            output = Array(T, length(first_result), xlen)

            for j in 1:N
                m = j
                for n in 1:size(output,1)
                    @inbounds output[n,m] = grad(first_result[n], j)
                end
                @inbounds gradvec[m] = G(x[m], gradzeros)
            end

            # Perform the rest of the chunks, filling in the output matrix
            for i in (N+1):N:xlen
                offset = i-1

                @simd for j in 1:N
                    m = j+offset
                    @inbounds gradvec[m] = G(x[m], partials[j])
                end
                
                chunk_result::Vector{ResultType} = f(gradvec)

                for j in 1:N
                    m = j+offset
                    for n in 1:size(output,1)
                        @inbounds output[n,m] = grad(chunk_result[n], j)
                    end
                    @inbounds gradvec[m] = G(x[m], gradzeros)
                end
            end
        end
    end

    return quote
        G = $G
        N = npartials(G)
        ResultType = G

        gradvec = get_workvec!(cache, GradientNumber, T, X, C)
        partials = get_partials!(cache, G)
        
        $body

        return output::Matrix{T}
    end
end