##################
# Taking Tensors #
##################
TensorCache() = ForwardDiffCache(TensorNumber)
const tens_void_cache = void_cache(TensorNumber)

# Exposed API methods #
#---------------------#
function tensor!{T}(output::Array{T,3},
                    f,
                    x::Vector;
                    chunk_size::Int=default_chunk,
                    cache::ForwardDiffCache=tens_void_cache)
    xlen = length(x)
    @assert (xlen, xlen, xlen) == size(output) "The output array must have size (length(input), length(input), length(input))"
    return _take_tensor!(output, f, x, chunk_size, cache)::Array{T,3}
end

function tensor{T}(f,
                   x::Vector{T};
                   chunk_size::Int=default_chunk,
                   cache::ForwardDiffCache=tens_void_cache)
    xlen = length(x)
    output = similar(x, xlen, xlen, xlen)
    return _take_tensor!(output, f, x, chunk_size, cache)::Array{T,3}
end

function tensor(f; mutates=false)
    cache = TensorCache()
    if mutates
        function tensf!{T}(output::Array{T,3}, x::Vector; chunk_size::Int=default_chunk)
            return tensor!(output, f, x, chunk_size=chunk_size, cache=cache)::Array{T,3}
        end
        return tensf!
    else
        function tensf{T}(x::Vector{T}; chunk_size::Int=default_chunk)
            return tensor(f, x, chunk_size=chunk_size, cache=cache)::Array{T,3}
        end
        return tensf
    end
end

# Calculate third order Taylor series term of a given function #
#--------------------------------------------------------------#
gradnum_type{N,T,C}(::Vector{TensorNumber{N,T,C}}) = GradientNumber{N,T,C}
hessnum_type{N,T,C}(::Vector{TensorNumber{N,T,C}}) = HessianNumber{N,T,C}

function _take_tensor!{T}(output::Array{T,3}, f, x::Vector, chunk_size::Int, cache::ForwardDiffCache)
    tensvec = get_workvec!(cache, x, chunk_size)
    partials = get_partials!(cache, eltype(tensvec))
    hesszeros = get_zeros!(cache, hessnum_type(tensvec))
    tenszeros = get_zeros!(cache, eltype(tensvec))
    if chunk_size_matches_full(x, chunk_size)
        return _calc_tensor_full!(output, f, x, tensvec, 
                                  partials, tenszeros, 
                                  hesszeros)::Array{T,3}
    else
        gradzeros = get_zeros!(cache, gradnum_type(tensvec))
        return _calc_tensor_chunks!(output, f, x, tensvec, 
                                    partials, tenszeros, 
                                    hesszeros, gradzeros)::Array{T,3}
    end
end

function _calc_tensor_full!{S,N,T,C}(output::Array{S,3},
                                     f,
                                     x::Vector{T},
                                     tensvec::Vector{TensorNumber{N,T,C}},
                                     partials, tenszeros, hesszeros)
    G = gradnum_type(tensvec)
    H = hessnum_type(tensvec)

    @simd for i in eachindex(x)
        @inbounds tensvec[i] = TensorNumber(H(G(x[i], partials[i]), hesszeros), tenszeros)
    end

    result = f(tensvec)

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

    return output::Array{S,3}
end

function _calc_tensor_chunks!{S,N,T,C}(output::Array{S,3},
                                       f,
                                       x::Vector{T},
                                       tensvec::Vector{TensorNumber{N,T,C}},
                                       partials, tenszeros, hesszeros, gradzeros)
    error("chunk_size configuration for ForwardDiff.tensor is not yet supported")
end
