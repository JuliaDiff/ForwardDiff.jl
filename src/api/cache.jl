############################################
# Methods for building work vectors/tuples #
############################################
typealias GradNumVec{N,T} GradientNumber{N,T,Vector{T}}
typealias GradNumTup{N,T} GradientNumber{N,T,NTuple{N,T}}

@generated function workvec_eltype{F,T,xlen,chunk_size}(::Type{F}, ::Type{T},
                                                        ::Type{Val{xlen}},
                                                        ::Type{Val{chunk_size}})
    if chunk_size == default_chunk_size
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

partials_type{N,T,C}(::Type{GradientNumber{N,T,C}}) = Partials{T,C}
partials_type{N,T,C}(::Type{HessianNumber{N,T,C}}) = partials_type(GradientNumber{N,T,C})
partials_type{N,T,C}(::Type{TensorNumber{N,T,C}}) = partials_type(GradientNumber{N,T,C})

function build_partials{N,T}(::Type{GradNumVec{N,T}})
    chunk_arr = Array(Partials{T,Vector{T}}, N)
    @simd for i in eachindex(chunk_arr)
        @inbounds chunk_arr[i] = Partials(setindex!(zeros(T, N), one(T), i))
    end
    return chunk_arr
end

function build_partials{N,T}(::Type{GradNumTup{N,T}})
    z = zero(T)
    o = one(T)
    partials_chunk = Vector{Partials{T, NTuple{N,T}}}(N)
    @simd for i in eachindex(partials_chunk)
        @inbounds partials_chunk[i] = Partials(ntuple(x -> ifelse(x == i, o, z), Val{N}))
    end
    return partials_chunk
end

build_partials{N,T,C}(::Type{HessianNumber{N,T,C}}) = build_partials(GradientNumber{N,T,C})
build_partials{N,T,C}(::Type{TensorNumber{N,T,C}}) = build_partials(GradientNumber{N,T,C})

zeros_type{N,T,C}(::Type{GradientNumber{N,T,C}}) = Partials{T,C}
zeros_type{N,T,C}(::Type{HessianNumber{N,T,C}}) = Vector{T}
zeros_type{N,T,C}(::Type{TensorNumber{N,T,C}}) = Vector{T}

build_zeros{N,T}(::Type{GradNumVec{N,T}}) = Partials(zeros(T, N))
build_zeros{N,T}(::Type{GradNumTup{N,T}}) = Partials(zero_tuple(NTuple{N,T}))
build_zeros{N,T,C}(::Type{HessianNumber{N,T,C}}) = zeros(T, halfhesslen(N))
build_zeros{N,T,C}(::Type{TensorNumber{N,T,C}}) = zeros(T, halftenslen(N))

#######################
# Cache Types/Methods #
#######################
immutable ForwardDiffCache
    workvec_cache::Function
    partials_cache::Function
    zeros_cache::Function
end

function ForwardDiffCache()
    const workvec_dict = Dict()
    const partials_dict = Dict()
    const zeros_dict = Dict()
    workvec_cache(args...) = cache_retrieve!(workvec_dict, build_workvec, args...)
    partials_cache(args...) = cache_retrieve!(partials_dict, build_partials, args...)
    zeros_cache(args...) = cache_retrieve!(zeros_dict, build_zeros, args...)
    return ForwardDiffCache(workvec_cache, partials_cache, zeros_cache)
end

function make_dummy_cache()
    workvec_cache(args...) = build_workvec(args...)
    partials_cache(args...) = build_partials(args...)
    zeros_cache(args...) = build_zeros(args...)
    return ForwardDiffCache(workvec_cache, partials_cache, zeros_cache)
end

function cache_retrieve!(dict, build_func, args...)
    if haskey(dict, args)
        return dict[args]
    else
        item = build_func(args...)
        dict[args] = item
        return item
    end
end

# Retrieval methods #
#-------------------#
function get_workvec!{F,T,xlen,chunk_size}(cache::ForwardDiffCache,
                                           ::Type{F}, ::Type{T},
                                           X::Type{Val{xlen}}, C::Type{Val{chunk_size}})
    return cache.workvec_cache(F, T, X, C)::Vector{workvec_eltype(F, T, X, C)}
end

function get_partials!{F}(cache::ForwardDiffCache, ::Type{F})
    return cache.partials_cache(F)::Vector{partials_type(F)}
end

function get_zeros!{F}(cache::ForwardDiffCache, ::Type{F})
    return cache.zeros_cache(F)::zeros_type(F)
end
