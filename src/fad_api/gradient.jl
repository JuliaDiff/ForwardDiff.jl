####################
# Taking Gradients #
####################

# Exposed API methods #
#---------------------#
function gradient!{T}(output::Vector{T},
                      f,
                      x::Vector;
                      chunk_size::Int=default_chunk,
                      cache::ForwardDiffCache=dummy_cache)
    xlen = length(x)
    @assert xlen == length(output) "The output vector must be the same length as the input vector"
    return _calc_gradient!(output, f, x, Val{xlen}, Val{chunk_size}, cache)::Vector{T}
end

function gradient{T}(f,
                     x::Vector{T};
                     chunk_size::Int=default_chunk,
                     cache::ForwardDiffCache=dummy_cache)
    return _calc_gradient!(similar(x), f, x, Val{length(x)}, Val{chunk_size}, cache)::Vector{T}
end

function gradient(f; mutates=false)
    cache = ForwardDiffCache()
    if mutates
        function gradf!{T}(output::Vector{T}, x::Vector; chunk_size::Int=default_chunk)
            return gradient!(output, f, x, chunk_size=chunk_size, cache=cache)::Vector{T}
        end
        return gradf!
    else
        function gradf{T}(x::Vector{T}; chunk_size::Int=default_chunk)
            return gradient(f, x, chunk_size=chunk_size, cache=cache)::Vector{T}
        end
        return gradf
    end
end
    
# Calculate gradient of a given function #
#----------------------------------------#
@generated function _calc_gradient!{S,T,xlen,chunk_size}(output::Vector{S}, f, x::Vector{T}, 
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

            result::ResultType = f(gradvec)

            @simd for i in eachindex(output)
                @inbounds output[i] = grad(result, i)
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
                    q = j+offset
                    @inbounds gradvec[q] = G(x[q], partials[j])
                end

                chunk_result::ResultType = f(gradvec)

                @simd for j in 1:N
                    q = j+offset
                    @inbounds output[q] = grad(chunk_result, j)
                    @inbounds gradvec[q] = G(x[q], gradzeros)
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

        return output::Vector{S}
    end
end