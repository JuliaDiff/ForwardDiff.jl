####################
# Taking Gradients #
####################

# Gradient Cache #
#----------------#
type GradientCache{T, Q, R}
    workvec::Vector{T}
    zeros::Partials{R}
    partials::Vector{Q}
end

function GradientCache{T}(input_size::Int, chunk_size::Int, Tv::Type{T}=Float64)
    G = GradientNumber{chunk_size, Tv, NTuple{chunk_size, Tv}}
    workvec = zeros(G, input_size)
    _zeros = build_zeros(G)
    partials = build_partials(G)
    return GradientCache(workvec, _zeros, partials)
end

get_workvec(cache::GradientCache) = cache.workvec
get_partials(cache::GradientCache) = cache.partials
get_zeros(cache::GradientCache) = cache.zeros

typealias GradForwardDiffCache Union{GradientCache, ForwardDiffCache}


# Exposed API methods #
#---------------------#
@generated function gradient!{T,A}(output::Vector{T}, f, x::Vector, ::Type{A}=Void;
                                   chunk_size::Int=default_chunk_size,
                                   cache::GradForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(gradient!(output, result)::Vector{T})
    elseif A <: AllResults
        return_stmt = :(gradient!(output, result)::Vector{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.gradient")
    end

    return quote
        result = _calc_gradient(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

@generated function gradient{T,A}(f, x::Vector{T}, ::Type{A}=Void;
                                  chunk_size::Int=default_chunk_size,
                                  cache::GradForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(gradient(result)::Vector{T})
    elseif A <: AllResults
        return_stmt = :(gradient(result)::Vector{T}, result)
    else
        error("invalid argument $A passed to FowardDiff.gradient")
    end

    return quote
        result = _calc_gradient(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

function gradient{A, T}(f, input_size::Int, input_type::Type{T},
                        ::Type{A}=Void;
                        mutates::Bool=false,
                        chunk_size::Int=default_chunk_size)
    if chunk_size == 0
        gcache = GradientCache(input_size, input_size, input_type)
    else
        gcache = GradientCache(input_size, chunk_size, input_type)
    end
    if mutates
        function g!(output::Vector, x::Vector)
            return ForwardDiff.gradient!(output, f, x, A;
                                         chunk_size=chunk_size,
                                         cache=gcache)
        end
        return g!
    else
        function g(x::Vector)
            return ForwardDiff.gradient(f, x, A;
                                        chunk_size=chunk_size,
                                        cache=gcache)
        end
        return g
    end
end

function gradient{A}(f, ::Type{A}=Void;
                     mutates::Bool=false,
                     chunk_size::Int=default_chunk_size,
                     cache::GradForwardDiffCache=ForwardDiffCache())
    if mutates
        function g!(output::Vector, x::Vector)
            return ForwardDiff.gradient!(output, f, x, A;
                                         chunk_size=chunk_size,
                                         cache=cache)
        end
        return g!
    else
        function g(x::Vector)
            return ForwardDiff.gradient(f, x, A;
                                        chunk_size=chunk_size,
                                        cache=cache)
        end
        return g
    end
end

# Calculate gradient of a given function #
#----------------------------------------#
function _calc_gradient{S}(f, x::Vector, ::Type{S},
                           chunk_size::Int,
                           cache::GradForwardDiffCache)
    X = Val{length(x)}
    C = Val{chunk_size}
    return _calc_gradient(f, x, S, X, C, cache)
end

@generated function _calc_gradient{T,S,xlen,chunk_size}(f, x::Vector{T}, ::Type{S},
                                                        X::Type{Val{xlen}},
                                                        C::Type{Val{chunk_size}},
                                                        cache::GradForwardDiffCache)
    check_chunk_size(xlen, chunk_size)
    G = workvec_eltype(GradientNumber, T, Val{xlen}, Val{chunk_size})
    if chunk_size_matches_vec_mode(xlen, chunk_size)
        # Vector-Mode
        ResultType = switch_eltype(G, S)
        body = quote
            @simd for i in 1:xlen
                @inbounds gradvec[i] = G(x[i], partials[i])
            end

            result::$ResultType = f(gradvec)
        end
    else
        # Chunk-Mode
        ChunkType = switch_eltype(G, S)
        ResultType = GradientNumber{xlen,S,Vector{S}}
        if cache <: GradientCache
            zero_body = quote gradzeros = get_zeros(cache) end
        else
            zero_body = quote gradzeros = get_zeros!(cache, G) end
        end

        body = quote
            $zero_body
            output = Vector{S}(xlen)

            @simd for i in 1:xlen
                @inbounds gradvec[i] = G(x[i], gradzeros)
            end

            local chunk_result::$ChunkType

            for i in 1:chunk_size:xlen
                offset = i-1

                @simd for j in 1:chunk_size
                    q = j+offset
                    @inbounds gradvec[q] = G(x[q], partials[j])
                end

                chunk_result = f(gradvec)

                @simd for j in 1:chunk_size
                    q = j+offset
                    @inbounds output[q] = grad(chunk_result, j)
                    @inbounds gradvec[q] = G(x[q], gradzeros)
                end
            end

            result::$ResultType = ($ResultType)(value(chunk_result), output)
        end
    end

    if cache <: GradientCache
        cache_body = quote
            gradvec = get_workvec(cache)
            partials = get_partials(cache)
            @assert length(gradvec) == xlen
            @assert eltype(gradvec[1]) == T
        end
    else
        cache_body = quote
            gradvec = get_workvec!(cache, GradientNumber, T, X, C)
            partials = get_partials!(cache, G)
        end
    end


    return quote
        G = $G
        $cache_body

        $body

        return ForwardDiffResult(result)
    end
end
