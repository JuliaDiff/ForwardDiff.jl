#########
# types #
#########

immutable DummyVar end

abstract ForwardDiffResult

######################
# parameter handling #
######################

@inline pickresult(::Val{false}, result::ForwardDiffResult, out) = out
@inline pickresult(::Val{true}, result::ForwardDiffResult, out) = result

const AUTO_CHUNK_THRESHOLD = 10

@generated function pickchunk{L}(len::Val{L})
    if L <= AUTO_CHUNK_THRESHOLD
        return :len
    else
        # Constrained to chunk <= AUTO_CHUNK_THRESHOLD, minimize (in order of priority):
        #   1. the number of chunks that need to be computed
        #   2. the number of "left over" perturbations in the final chunk
        nchunks = round(Int, L / AUTO_CHUNK_THRESHOLD, RoundUp)
        C = round(Int, L / nchunks, RoundUp)
        return :(Val{$C}())
    end
end

@noinline pickchunk(chunk::Val{nothing}, ::Val{nothing}, x) = pickchunk(chunk, Val{length(x)}(), x)
@noinline pickchunk{C}(chunk::Val{C}, ::Val{nothing}, x) = pickchunk(chunk, Val{length(x)}(), x)
@noinline pickchunk{L}(chunk::Val{nothing}, len::Val{L}, x) = pickchunk(pickchunk(len), len, x)
@noinline pickchunk{C,L}(chunk::Val{C}, len::Val{L}, x) = (chunk, len)

###################
# macro utilities #
###################

const KWARG_DEFAULTS = (:chunk => nothing, :len => nothing, :allresults => false, :multithread => false)

iskwarg(ex) = isa(ex, Expr) && (ex.head == :kw || ex.head == :(=))

function separate_kwargs(args)
    # if called as `f(args...; kwargs...)`, i.e. with a semicolon
    if isa(first(args), Expr) && first(args).head == :parameters
        kwargs = first(args).args
        args = args[2:end]
    else # if called as `f(args..., kwargs...)`, i.e. with a comma
        i = findfirst(iskwarg, args)
        if i == 0
            kwargs = tuple()
        else
            kwargs = args[i:end]
            args = args[1:i-1]
        end
    end
    return args, kwargs
end

function arrange_kwargs(kwargs, defaults)
    keys = map(first, defaults)
    badargs = setdiff(map(kw -> kw.args[1], kwargs), keys)
    @assert isempty(badargs) "unrecognized keyword arguments: $(badargs)"
    return [:(Val{$(getkw(kwargs, key, defaults))}()) for key in keys]
end

function getkw(kwargs, key, defaults)
    for kwexpr in kwargs
        if kwexpr.args[1] == key
            return kwexpr.args[2]
        end
    end
    return default_value(defaults, key)
end

function default_value(defaults, key)
    for kwpair in defaults
        if kwpair.first == key
            return kwpair.second
        end
    end
    throw(KeyError(key))
end

#######################
# caching work arrays #
#######################

const CACHE = ntuple(n -> Dict{DataType,Any}(), NTHREADS)

function clearcache!()
    for d in CACHE
        empty!(d)
    end
end

@eval cachefetch!{D,L}(::Type{D}, len::Val{L}) = $(Expr(:tuple, [:(cachefetch!($i, D, len)) for i in 1:NTHREADS]...))

function cachefetch!{T,L}(tid::Integer, ::Type{T}, ::Val{L})
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

function cachefetch!{N,T,Z}(tid::Integer, ::Type{Partials{N,T}}, ::Val{Z})
    K = Tuple{Partials{N,T},Z}
    V = Vector{Partials{N,T}}
    cache = CACHE[tid]
    if haskey(cache, K)
        seeds = cache[K]::V
    else
        seeds = V(Z)
        x = one(T)
        for i in 1:Z
            seeds[i] = setindex(zero(Partials{N,T}), x, i)
        end
        cache[K] = seeds
    end
    return seeds::V
end

function cachefetch!{N,T}(tid::Integer, ::Type{Partials{N,T}})
    return cachefetch!(tid, Partials{N,T}, Val{N}())
end

function threaded_fetchxdiff{N,L}(x, chunk::Val{N}, len::Val{L})
    return cachefetch!(Dual{N,eltype(x)}, len)
end

function fetchxdiff{N,L}(x, chunk::Val{N}, len::Val{L})
    return cachefetch!(compat_threadid(), Dual{N,eltype(x)}, len)
end

function fetchseeds{N,T}(::Vector{Dual{N,T}}, args...)
    return cachefetch!(compat_threadid(), Partials{N,T}, args...)
end

#######################
# seeding work arrays #
#######################

function seedall!{N,T,L}(xdiff::Vector{Dual{N,T}}, x, len::Val{L}, seed::Partials{N,T})
    @simd for i in 1:L
        @inbounds xdiff[i] = Dual{N,T}(x[i], seed)
    end
    return xdiff
end

function seed!{N,T}(xdiff::Vector{Dual{N,T}}, x, seed::Partials{N,T}, offset)
    k = offset - 1
    @simd for i in 1:N
        j = i + k
        @inbounds xdiff[j] = Dual{N,T}(x[j], seed)
    end
    return xdiff
end

function seed!{N,T}(xdiff::Vector{Dual{N,T}}, x, seeds::Vector{Partials{N,T}}, offset)
    k = offset - 1
    @simd for i in 1:N
        j = i + k
        @inbounds xdiff[j] = Dual{N,T}(x[j], seeds[i])
    end
    return xdiff
end
