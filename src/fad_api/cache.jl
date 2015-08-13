############################################
# Methods for building work vectors/tuples #
############################################
@generated function workvec_eltype{F,T,xlen,chunk_size}(::Type{F}, ::Type{T}, 
                                                        ::Type{Val{xlen}}, 
                                                        ::Type{Val{chunk_size}})
    if chunk_size == default_chunk
        C = xlen > tuple_usage_threshold ? Vector{T} : NTuple{xlen,T}
        return :($F{$xlen,$T,$C})
    else
        return :($F{$chunk_size,$T,NTuple{$chunk_size,$T}})
    end
end

@generated function build_workvec{F,T,xlen,chunk_size}(::Type{F}, ::Type{T}, 
                                                       ::Type{Val{xlen}}, 
                                                       ::Type{Val{chunk_size}})
    G = workvec_eltype(F, T, Val{xlen}, Val{chunk_size})
    return :(Vector{$G}($xlen))
end

partials_type{N,T}(::Type{GradNumVec{N,T}}) = Vector{Vector{T}}

@generated function partials_type{N,T}(::Type{GradNumTup{N,T}})
    if N > tuple_usage_threshold
        ex = :(Vector{NTuple{$N,$T}})
    else
        ex = :(NTuple{$N,NTuple{$N,$T}})
    end
    return ex
end

partials_type{N,T,C}(::Type{HessianNumber{N,T,C}}) = partials_type(GradientNumber{N,T,C})
partials_type{N,T,C}(::Type{TensorNumber{N,T,C}}) = partials_type(GradientNumber{N,T,C})

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

zeros_type{N,T}(::Type{GradNumVec{N,T}}) = Vector{T}
zeros_type{N,T}(::Type{GradNumTup{N,T}}) = NTuple{N,T}
zeros_type{N,T,C}(::Type{HessianNumber{N,T,C}}) = Vector{T}
zeros_type{N,T,C}(::Type{TensorNumber{N,T,C}}) = Vector{T}

#######################
# Cache Types/Methods #
#######################
immutable ForwardDiffCache
    workvec_cache::Function
    partials_cache::Function
    zeros_cache::Function
end

function ForwardDiffCache()
    @generated function workvec_cache{F,T,xlen,chunk_size}(::Type{F},::Type{T}, 
                                                           ::Type{Val{xlen}}, 
                                                           ::Type{Val{chunk_size}})
        result = build_workvec(F, T, Val{xlen}, Val{chunk_size})
        return :($result)
    end
    @generated function partials_cache{F}(::Type{F})
        result = build_partials(F)
        return :($result)
    end
    @generated function zeros_cache{F}(::Type{F})
        result = build_zeros(F)
        return :($result)
    end
    return ForwardDiffCache(workvec_cache, partials_cache, zeros_cache)
end

function make_void_cache()
    workvec_cache(args...) = build_workvec(args...) 
    partials_cache(args...) = build_partials(args...)
    zeros_cache(args...) = build_zeros(args...)
    return ForwardDiffCache(workvec_cache, partials_cache, zeros_cache)
end

# Retrieval methods #
#-------------------#
function get_workvec!{F,T,xlen,chunk_size}(cache::ForwardDiffCache,
                                           ::Type{F}, ::Type{T},
                                           X::Type{Val{xlen}}, C::Type{Val{chunk_size}})
    return cache.workvec_cache(F, T, X, C)::Vector{workvec_eltype(F, T, X, C)}
end

function get_partials!{F}(cache::ForwardDiffCache, ::Type{F})
    return cache.partials_cache(F)::partials_type(F)
end

function get_zeros!{F}(cache::ForwardDiffCache, ::Type{F})
    return cache.zeros_cache(F)::zeros_type(F)
end
