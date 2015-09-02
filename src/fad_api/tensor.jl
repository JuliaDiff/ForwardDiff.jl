##################
# Taking Tensors #
##################

# Exposed API methods #
#---------------------#
@generated function tensor!{T,A}(output::Array{T,3}, f, x::Vector, ::Type{A}=Void;
                                 chunk_size::Int=default_chunk_size,
                                 cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(tensor!(output, result)::Array{T,3})
    elseif A <: AllResults
        return_stmt = :(tensor!(output, result)::Array{T,3}, result)
    else
        error("invalid argument $A passed to FowardDiff.tensor")
    end

    return quote
        result = _calc_tensor(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

@generated function tensor{T,A}(f, x::Vector{T}, ::Type{A}=Void;
                                chunk_size::Int=default_chunk_size,
                                cache::ForwardDiffCache=dummy_cache)
    if A <: Void
        return_stmt = :(tensor(result)::Array{T,3})
    elseif A <: AllResults
        return_stmt = :(tensor(result)::Array{T,3}, result)
    else
        error("invalid argument $A passed to FowardDiff.tensor")
    end

    return quote
        result = _calc_tensor(f, x, T, chunk_size, cache)
        return $return_stmt
    end
end

function tensor{A}(f, ::Type{A}=Void;
                    mutates::Bool=false,
                    chunk_size::Int=default_chunk_size,
                    cache::ForwardDiffCache=ForwardDiffCache())
    if mutates
        function t!(output::Array, x::Vector)
            return ForwardDiff.tensor!(output, f, x, A;
                                       chunk_size=chunk_size,
                                       cache=cache)
        end
        return t!
    else
        function t(x::Vector)
            return ForwardDiff.tensor(f, x, A;
                                      chunk_size=chunk_size,
                                      cache=cache)
        end
        return t
    end
end

# Calculate third order Taylor series term of a given function #
#--------------------------------------------------------------#
function _calc_tensor{S}(f, x::Vector, ::Type{S},
                         chunk_size::Int,
                         cache::ForwardDiffCache)
    X = Val{length(x)}
    C = Val{chunk_size}
    return _calc_tensor(f, x, S, X, C, cache)
end

hessnum_type{N,T,C}(::Type{TensorNumber{N,T,C}}) = HessianNumber{N,T,C}

@generated function _calc_tensor{T,S,xlen,chunk_size}(f, x::Vector{T}, ::Type{S},
                                                      X::Type{Val{xlen}},
                                                      C::Type{Val{chunk_size}},
                                                      cache::ForwardDiffCache)
    check_chunk_size(xlen, chunk_size)

    F = workvec_eltype(TensorNumber, T, Val{xlen}, Val{chunk_size})
    H = hessnum_type(F)
    G = gradnum_type(H)

    if chunk_size_matches_vec_mode(xlen, chunk_size)
        # Vector-Mode
        ResultType = switch_eltype(F, S)
        body = quote
            @simd for i in eachindex(x)
                @inbounds tensvec[i] = TensorNumber(H(G(x[i], partials[i]), hesszeros), tenszeros)
            end

            result::$ResultType = f(tensvec)
        end
    else
        error("chunk_size configuration for ForwardDiff.tensor is not yet supported")
    end

    return quote
        F, H, G = $F, $H, $G
        tensvec = get_workvec!(cache, TensorNumber, T, X, C)
        partials = get_partials!(cache, F)
        tenszeros = get_zeros!(cache, F)
        hesszeros = get_zeros!(cache, H)

        $body

        return ForwardDiffResult(result)
    end
end
