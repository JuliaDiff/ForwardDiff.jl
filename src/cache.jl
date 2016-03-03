#################
# GradientCache #
#################

immutable GradientCache{chunk, T}
    workvec::Vector{DiffNumber{chunk, T}}
    partials::Vector{Partials{chunk, T}}
    partials_remainder::Vector{Partials{chunk, T}}
end

totalsizeof(GC::GradientCache) = sizeof(GC.workvec) + sizeof(GC.partials) + sizeof(GC.partials_remainder)

function GradientCache{chunk, T}(remainder, input_length, ::Type{Val{chunk}}, ::Type{T})
    V_diffnumber = Vector{DiffNumber{chunk, T}}(input_length)
    V_partials = Vector{Partials{chunk,T}}(chunk)
    V_partials_remainder = Vector{Partials{chunk,T}}(remainder)

    x = one(T)
    for i in 1:chunk
        V_partials[i] = setindex(zero(Partials{chunk,T}), x, i)
        if i <= remainder
            V_partials_remainder[i] = setindex(zero(Partials{chunk,T}), x, i)
        end
    end

    return GradientCache{chunk, T}(V_diffnumber, V_partials, V_partials_remainder)
end


function cachefetch!{T,L}(tid::Integer, ::Type{T}, ::Type{Val{L}})
    K = Tuple{T,L}
    V = Vector{T}
    cache = CACHE[tid]
    if haskey(cache, K)
        v = cache[K]::V
    else
        v = V(L)
        cache[K] = v
    end
    return v
end


####################
# ForwardDiffCache #
####################

immutable ForwardDiffCache{chunk, T}
    caches::Vector{GradientCache{chunk,T}}
end

function Base.show{chunk, T}(io::IO, fdc::ForwardDiffCache{chunk, T})
    print(io, "ForwardDiffCache: length: $(get_input_length(fdc)), chunk: $(chunk), T: $(T)")
end

get_chunk_size{CHUNK, T}(fdc::ForwardDiffCache{CHUNK, T}) = CHUNK
get_type{CHUNK, T}(fdc::ForwardDiffCache{CHUNK, T}) = T
get_input_length(fdc::ForwardDiffCache) = length(fdc.caches[1].workvec)
get_cache(fdc::ForwardDiffCache) = fdc.caches[compat_threadid()]
totalsizeof(fdc::ForwardDiffCache) = sum(map(totalsizeof, fdc.caches))

@generated function ForwardDiffCache{input_length, T}(::Type{Val{input_length}}, ::Type{T})
    chunk = pick_chunk(input_length)
    return quote
        return ForwardDiffCache(Val{input_length}, T, Val{$chunk})
    end
end


@generated function ForwardDiffCache{input_length, chunk, T}(::Type{Val{input_length}}, ::Type{T}, ::Type{Val{chunk}})
    chunk_value = chunk == nothing ? pick_chunk(input_length) : chunk
    body = quote
        remainder = compute_remainder(input_length, $chunk_value)
        cache_vec = Vector{GradientCache{$chunk_value, T}}(NTHREADS)
        for i in 1:NTHREADS
            cache_vec[i] = GradientCache(remainder, input_length, Val{$chunk_value}, T)
        end
        return ForwardDiffCache(cache_vec)::ForwardDiffCache{$chunk_value, T}
    end
    return body
end

function generate_cache_body(caches, input_length, chunk)
    if caches <: Type{Void}
        cache_body = quote
            _caches = cachefetch!(Val{$input_length}, Val{$chunk}, T)
        end
    else
        cache_body = quote
            _caches = caches
        end
    end
    return cache_body
end


#########
# CACHE #
#########

const _CACHE = Dict{Tuple{Int, Int, DataType}, ForwardDiffCache}()

function clearcache!()
    empty!(CACHE)
    return
end

# Computes the total size in bytes of the cache
totalsizeofcache() = sum(map(totalsizeof, values(_CACHE)))

function cachefetch!{input_length, chunk, T}(::Type{Val{input_length}}, ::Type{Val{chunk}}, ::Type{T})
    K = (input_length, chunk, T)
    if haskey(_CACHE, K)
        caches = _CACHE[K]
    else
        caches = ForwardDiffCache(Val{input_length}, T, Val{chunk})
        _CACHE[K] = caches
    end
    return caches::ForwardDiffCache{chunk, T}
end
