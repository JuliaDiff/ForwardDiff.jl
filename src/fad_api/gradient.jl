####################
# Taking Gradients #
####################
GradientCache() = ForwardDiffCache(GradientNumber)
const grad_void_cache = void_cache(GradientNumber)

# Exposed API methods #
#---------------------#
function gradient!{T}(output::Vector{T},
                      f,
                      x::Vector;
                      chunk_size::Int=default_chunk,
                      cache::ForwardDiffCache=grad_void_cache)
    @assert length(x) == length(output) "The output vector must be the same length as the input vector"
    return _take_gradient!(output, f, x, chunk_size, cache)::Vector{T}
end

function gradient{T}(f,
                     x::Vector{T};
                     chunk_size::Int=default_chunk,
                     cache::ForwardDiffCache=grad_void_cache)
    return _take_gradient!(similar(x), f, x, chunk_size, cache)::Vector{T}
end

function gradient(f; mutates=false)
    cache = GradientCache()
    if mutates
        function gradf!(output::Vector, x::Vector; chunk_size::Int=default_chunk)
            return gradient!(output, f, x, chunk_size=chunk_size, cache=cache)
        end
        return gradf!
    else
        function gradf(x::Vector; chunk_size::Int=default_chunk)
            return gradient(f, x, chunk_size=chunk_size, cache=cache)
        end
        return gradf
    end
end
    
# Calculate gradient of a given function #
#----------------------------------------#
function _take_gradient!(output::Vector, f, x::Vector, chunk_size::Int, cache::ForwardDiffCache)
    gradvec = get_workvec!(cache, x, chunk_size)
    partials = get_partials!(cache, eltype(gradvec))
    if chunk_size_matches_full(x, chunk_size)
        return _calc_gradient_full!(output, f, x, gradvec, partials)
    else
        gradzeros = get_zeros!(cache, eltype(gradvec))
        return _calc_gradient_chunks!(output, f, x, gradvec, partials, gradzeros)
    end
end

function _calc_gradient_full!{S,N,T,C}(output::Vector{S},
                                       f,
                                       x::Vector{T},
                                       gradvec::Vector{GradientNumber{N,T,C}},
                                       partials)
    _load_gradvec_with_x_partials!(gradvec, x, partials)

    result = f(gradvec)

    @simd for i in eachindex(output)
        @inbounds output[i] = grad(result, i)
    end                     

    return output::Vector{S}
end

function _calc_gradient_chunks!{S,N,T,C}(output::Vector{S},
                                         f,
                                         x::Vector{T},
                                         gradvec::Vector{GradientNumber{N,T,C}},
                                         partials, gradzeros)
    xlen = length(x)
    G = eltype(gradvec)

    check_chunk_size(x, N)

    _load_gradvec_with_x_zeros!(gradvec, x, gradzeros)

    for i in 1:N:xlen
        _load_gradvec_with_x_partials!(gradvec, x, partials, i)
        
        chunk_result = f(gradvec)

        # load resultant partials components
        # into output, replacing them with 
        # zeros in gradvec
        @simd for j in 1:N
            q = i+j-1
            @inbounds output[q] = grad(chunk_result, j)
            @inbounds gradvec[q] = G(x[q], gradzeros)
        end
    end

    return output::Vector{S}
end

# Helper functions #
#------------------#
function _load_gradvec_with_x_partials!(gradvec, x, partials)
    # fill gradvec with GradientNumbers of single partial components
    G = eltype(gradvec)
    @simd for i in eachindex(gradvec)
        @inbounds gradvec[i] = G(x[i], partials[i])
    end
end

function _load_gradvec_with_x_partials!{N,T,C}(gradvec::Vector{GradientNumber{N,T,C}}, x, partials, init)
    # fill current chunk gradvec with GradientNumbers of single partial components
    G = eltype(gradvec)
    i = init-1
    @simd for j in 1:N
        m = i+j
        @inbounds gradvec[m] = G(x[m], partials[j])
    end
end

function _load_gradvec_with_x_zeros!(gradvec, x, gradzeros)
    # fill gradvec with x[i]-valued GradientNumbers
    G = eltype(gradvec)
    @simd for i in eachindex(x)
        @inbounds gradvec[i] = G(x[i], gradzeros)
    end
end