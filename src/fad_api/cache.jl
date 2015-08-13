############################################
# Methods for building work vectors/tuples #
############################################
@generated function build_workvec{F,T,xlen,chunk_size}(::Type{F}, ::Type{T}, 
                                                       ::Type{Val{xlen}}, 
                                                       ::Type{Val{chunk_size}})
    if chunk_size == default_chunk
        C = xlen > tuple_usage_threshold ? Vector{T} : NTuple{xlen,T}
        return :(Vector{$F{$xlen,$T,$C}}($xlen))
    else
        return :(Vector{$F{$chunk_size,$T,NTuple{$chunk_size,$T}}}($xlen))
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
immutable ForwardDiffCache{F}
    fad_type::Type{F}
    workvec_cache::Function
    partials_cache::Function
    zeros_cache::Function
end

function ForwardDiffCache{F}(::Type{F})
    @generated function workvec_cache{T,xlen,chunk_size}(::Type{T}, 
                                                         ::Type{Val{xlen}}, 
                                                         ::Type{Val{chunk_size}})
        result = build_workvec(F, T, Val{xlen}, Val{chunk_size})
        return :($result)
    end
    @generated function partials_cache{G}(::Type{G})
        result = build_partials(G)
        return :($result)
    end
    @generated function zeros_cache{G}(::Type{G})
        result = build_zeros(G)
        return :($result)
    end
    return ForwardDiffCache(F, workvec_cache, partials_cache, zeros_cache)
end

function void_cache{F}(::Type{F})
    workvec_cache(args...) = build_workvec(F, args...) 
    partials_cache(args...) = build_partials(args...)
    zeros_cache(args...) = build_zeros(args...)
    return ForwardDiffCache(F, workvec_cache, partials_cache, zeros_cache)
end

# Retrieval methods #
#-------------------#
function get_workvec!{T}(cache::ForwardDiffCache, x::Vector{T}, chunk_size::Int)
    return cache.workvec_cache(T, Val{length(x)}, Val{chunk_size})
end

function get_partials!{G}(cache::ForwardDiffCache, ::Type{G})
    return cache.partials_cache(G)
end

function get_zeros!{G}(cache::ForwardDiffCache, ::Type{G})
    return cache.zeros_cache(G)
end
