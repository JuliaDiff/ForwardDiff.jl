##################
# Taking Tensors #
##################

# Exposed API methods #
#---------------------#
function tensor!{T}(output::Array{T,3},
                    f,
                    x::Vector;
                    chunk_size::Int=default_chunk_size,
                    cache::ForwardDiffCache=dummy_cache)
    xlen = length(x)
    @assert (xlen, xlen, xlen) == size(output) "The output array must have size (length(input), length(input), length(input))"
    return _calc_tensor!(output, f, x, Val{xlen}, Val{chunk_size}, cache)::Array{T,3}
end

function tensor{T}(f,
                   x::Vector{T};
                   chunk_size::Int=default_chunk_size,
                   cache::ForwardDiffCache=dummy_cache)
    xlen = length(x)
    output = similar(x, xlen, xlen, xlen)
    return _calc_tensor!(output, f, x, Val{xlen}, Val{chunk_size}, cache)::Array{T,3}
end

function tensor(f; mutates=false)
    cache = ForwardDiffCache()
    if mutates
        function tensf!{T}(output::Array{T,3}, x::Vector; chunk_size::Int=default_chunk_size)
            return ForwardDiff.tensor!(output, f, x, chunk_size=chunk_size, cache=cache)::Array{T,3}
        end
        return tensf!
    else
        function tensf{T}(x::Vector{T}; chunk_size::Int=default_chunk_size)
            return ForwardDiff.tensor(f, x, chunk_size=chunk_size, cache=cache)::Array{T,3}
        end
        return tensf
    end
end

# Calculate third order Taylor series term of a given function #
#--------------------------------------------------------------#
hessnum_type{N,T,C}(::Type{TensorNumber{N,T,C}}) = HessianNumber{N,T,C}

@generated function _calc_tensor!{S,T,xlen,chunk_size}(output::Array{S,3}, f, x::Vector{T}, 
                                                       X::Type{Val{xlen}}, C::Type{Val{chunk_size}},
                                                       cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)
    full_bool = chunk_size_matches_vec_mode(xlen, chunk_size)
    
    F = workvec_eltype(TensorNumber, T, Val{xlen}, Val{chunk_size})
    H = hessnum_type(F)
    G = gradnum_type(H)

    if full_bool
        body = quote
            @simd for i in eachindex(x)
                @inbounds tensvec[i] = TensorNumber(H(G(x[i], partials[i]), hesszeros), tenszeros)
            end

            result::ResultType = f(tensvec)

            q = 1
            for i in 1:N
                for j in i:N
                    for k in i:j
                        @inbounds output[j, k, i] = tens(result, q)::S
                        q += 1
                    end
                end
                for j in 1:(i-1)
                    for k in 1:j
                        @inbounds output[j, k, i] = output[i, j, k]
                    end
                end    
                for j in i:N
                    for k in 1:(i-1)
                        @inbounds output[j, k, i] = output[i, j, k]
                    end
                end
                for j in 1:N
                    for k in (j+1):N
                        @inbounds output[j, k, i] = output[k, j, i]
                    end
                end
            end
        end
    else
        error("chunk_size configuration for ForwardDiff.tensor is not yet supported")
    end

    return quote

        F = $F
        H = $H
        G = $G
        N = npartials(F)
        ResultType = switch_eltype(F, S)

        tensvec = get_workvec!(cache, TensorNumber, T, X, C)
        partials = get_partials!(cache, F)
        tenszeros = get_zeros!(cache, F)
        hesszeros = get_zeros!(cache, H)
        
        $body

        return output::Array{S,3}
    end
end
