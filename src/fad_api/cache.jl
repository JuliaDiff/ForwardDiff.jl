############################################
# Methods for building work vectors/tuples #
############################################
function build_workvec{F,T}(::Type{F}, x::Vector{T}, chunk_size::Int)
    xlen = length(x)
    if chunk_size == default_chunk
        C = xlen > tuple_usage_threshold ? Vector{T} : NTuple{xlen,T}
        return Vector{F{xlen,T,C}}(xlen)
    else
        return Vector{F{chunk_size,T,NTuple{chunk_size,T}}}(xlen)
    end
end

function build_partials{N,T}(::Type{GradNumVec{N,T}})
    chunk_arr = Array(Vector{T}, N)
    @simd for i in eachindex(chunk_arr)
        @inbounds chunk_arr[i] = setindex!(zeros(T, N), one(T), i)
    end
    return chunk_arr
end

@generated function build_partials{N,T}(::Type{GradNumTup{N,T}})
    if N > tuple_usage_threshold
        ex = quote
            partials_chunk = Vector{NTuple{$N,$T}}($N)
            @simd for i in eachindex(partials_chunk)
                @inbounds partials_chunk[i] = ntuple(x -> ifelse(x == i, o, z), Val{$N})
            end
            return partials_chunk
        end
    else
        ex = quote
            return ntuple(i -> ntuple(x -> ifelse(x == i, o, z), Val{$N}), Val{$N})
        end
    end
    return quote
        z = zero(T)
        o = one(T)
        $ex
    end
end

build_partials{N,T,C}(::Type{HessianNumber{N,T,C}}) = build_partials(GradientNumber{N,T,C})
build_partials{N,T,C}(::Type{TensorNumber{N,T,C}}) = build_partials(GradientNumber{N,T,C})

build_zeros{N,T}(::Type{GradNumVec{N,T}}) = zeros(T, N)
build_zeros{N,T}(::Type{GradNumTup{N,T}}) = zero_tuple(NTuple{N,T})
build_zeros{N,T,C}(::Type{HessianNumber{N,T,C}}) = zeros(T, halfhesslen(N))
build_zeros{N,T,C}(::Type{TensorNumber{N,T,C}}) = zeros(T, halftenslen(N))

#######################
# Cache Types/Methods #
#######################
type ForwardDiffCache{F,W,P,Z}
    fad_type::Type{F}
    workvecs::W
    partials::P
    zeros::Z
end

typealias VoidCache{F} ForwardDiffCache{F,Void,Void,Void}

ForwardDiffCache{F}(::Type{F}) = ForwardDiffCache(F, Dict(), Dict(), Dict())
void_cache{F}(::Type{F}) = ForwardDiffCache(F, nothing, nothing, nothing) 

# Retrieval methods #
#-------------------#
function get_workvec!{F,T}(cache::ForwardDiffCache{F}, x::Vector{T}, chunk_size::Int)
    key = tuple(length(x), T, chunk_size)
    if haskey(cache.workvecs, key)
        return cache.workvecs[key]
    else
        workvec = build_workvec(F, x, chunk_size)
        cache.workvecs[key] = workvec
        return workvec
    end
end

function get_workvec!{F,T}(cache::VoidCache{F}, x::Vector{T}, chunk_size::Int)
    return build_workvec(F, x, chunk_size)
end

function get_partials!{F}(cache::ForwardDiffCache, ::Type{F})
    if haskey(cache.partials, F)
        return cache.partials[F]
    else
        partials = build_partials(F)
        cache.partials[F] = partials
        return partials
    end
end

function get_partials!{F1,F2}(cache::VoidCache{F1}, ::Type{F2})
    return build_partials(F2)
end

function get_zeros!{F}(cache::ForwardDiffCache, ::Type{F})
    if haskey(cache.zeros, F)
        return cache.zeros[F]
    else
        zeros = build_zeros(F)
        cache.zeros[F] = zeros
        return zeros
    end
end

function get_zeros!{F1,F2}(cache::VoidCache{F1}, ::Type{F2})
    return build_zeros(F2)
end