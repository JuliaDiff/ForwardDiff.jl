####################
# Taking Jacobians #
####################
JacobianCache() = ForwardDiffCache(GradientNumber)
const jac_void_cache = void_cache(GradientNumber)

# Exposed API methods #
#---------------------#
function jacobian!{T}(output::Matrix{T},
                      f,
                      x::Vector;
                      chunk_size::Int=default_chunk,
                      cache::ForwardDiffCache=jac_void_cache)
    @assert length(x) == size(output, 2) "The output matrix must have a number of columns equal to the length of the input vector"
    return _take_jacobian!(output, f, x, chunk_size, cache)::Matrix{T}
end

function jacobian{T}(f,
                     x::Vector{T};
                     chunk_size::Int=default_chunk,
                     cache::ForwardDiffCache=jac_void_cache)
    return _take_jacobian(f, x, chunk_size, cache)::Matrix{T}
end

function jacobian(f; mutates=false)
    cache = JacobianCache()
    if mutates
        function jacf!(output::Matrix, x::Vector; chunk_size::Int=default_chunk)
            return jacobian!(output, f, x, chunk_size=chunk_size, cache=cache)
        end
        return jacf!
    else
        function jacf(x::Vector; chunk_size::Int=default_chunk)
            return jacobian(f, x, chunk_size=chunk_size, cache=cache)
        end
        return jacf
    end
end

# Calculate Jacobian of a given function #
#----------------------------------------#
function _take_jacobian!(output::Matrix, f, x::Vector, chunk_size::Int, cache::ForwardDiffCache)
    gradvec = get_workvec!(cache, x, chunk_size)
    partials = get_partials!(cache, eltype(gradvec))
    if chunk_size_matches_full(x, chunk_size)
        return _calc_jacobian_full!(output, f, x, gradvec, partials)
    else
        gradzeros = get_zeros!(cache, eltype(gradvec))
        return _calc_jacobian_chunks!(output, f, x, gradvec, partials, gradzeros)
    end
end

function _take_jacobian(f, x::Vector, chunk_size::Int, cache::ForwardDiffCache)
    gradvec = get_workvec!(cache, x, chunk_size)
    partials = get_partials!(cache, eltype(gradvec))
    if chunk_size_matches_full(x, chunk_size)
        return _calc_jacobian_full(f, x, gradvec, partials)
    else
        gradzeros = get_zeros!(cache, eltype(gradvec))
        return _calc_jacobian_chunks(f, x, gradvec, partials, gradzeros)
    end
end

function _calc_jacobian_full!{S,N,T,C}(output::Matrix{S},
                                       f,
                                       x::Vector{T},
                                       gradvec::Vector{GradientNumber{N,T,C}},
                                       partials)
    _load_gradvec_with_x_partials!(gradvec, x, partials)

    result = f(gradvec)

    for i in eachindex(result), j in eachindex(x)
        output[i,j] = grad(result[i], j)
    end

    return output::Matrix{S}
end

function _calc_jacobian_chunks!{S,N,T,C}(output::Matrix{S},
                                         f,
                                         x::Vector{T},
                                         gradvec::Vector{GradientNumber{N,T,C}},
                                         partials, gradzeros) 
    check_chunk_size(x, N)

    _load_gradvec_with_x_zeros!(gradvec, x, gradzeros)

    _fill_jac_chunks!(output, f, x, gradvec, partials, gradzeros, 1)

    return output::Matrix{S}
end

function _calc_jacobian_full{N,T,C}(f,
                                    x::Vector{T},
                                    gradvec::Vector{GradientNumber{N,T,C}},
                                    partials) 

    _load_gradvec_with_x_partials!(gradvec, x, partials)

    result = f(gradvec)
    output = Array(T, length(result), length(x))

    for i in eachindex(result), j in eachindex(x)
        output[i,j] = grad(result[i], j)
    end

    return output::Matrix{T}
end

function _calc_jacobian_chunks{N,T,C}(f,
                                      x::Vector{T},
                                      gradvec::Vector{GradientNumber{N,T,C}},
                                      partials, gradzeros) 
    xlen = length(x)
    G = eltype(gradvec)

    check_chunk_size(x, N)

    _load_gradvec_with_x_zeros!(gradvec, x, gradzeros)

    # Perform the first chunk "manually" so that
    # we get to inspect the size of the output
    i = 1

    _load_gradvec_with_x_partials!(gradvec, x, partials, i)

    chunk_result = f(gradvec)

    output = Array(T, length(chunk_result), xlen)

    for j in 1:N
        m = i+j-1
        for n in 1:size(output,1)
            @inbounds output[n,m] = grad(chunk_result[n], j)
        end
        @inbounds gradvec[m] = G(x[m], gradzeros)
    end

    # Now perform the rest of the chunks, filling in the output matrix
    _fill_jac_chunks!(output, f, x, gradvec, partials, gradzeros, N+1)

    return output::Matrix{T}
end

# Helper functions #
#------------------#
function _fill_jac_chunks!{N,T,C}(output,
                                  f,
                                  x::Vector{T}, 
                                  gradvec::Vector{GradientNumber{N,T,C}},
                                  partials, gradzeros, init)
    G = eltype(gradvec)

    for i in init:N:length(x)
        _load_gradvec_with_x_partials!(gradvec, x, partials, i)

        chunk_result = f(gradvec)

        for j in 1:N
            m = i+j-1
            for n in 1:size(output,1)
                @inbounds output[n,m] = grad(chunk_result[n], j)
            end
            @inbounds gradvec[m] = G(x[m], gradzeros)
        end
    end

    return output
end