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

@generated function pickchunk{L}(xlength::Val{L})
    if L <= AUTO_CHUNK_THRESHOLD
        return :xlength
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
@noinline pickchunk{L}(chunk::Val{nothing}, xlength::Val{L}, x) = pickchunk(pickchunk(xlength), xlength, x)
@noinline pickchunk{C,L}(chunk::Val{C}, xlength::Val{L}, x) = (chunk, xlength)

###################
# macro utilities #
###################

const KWARG_DEFAULTS = (:allresults => false, :chunk => nothing, :multithread => false, :xlength => nothing)

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

function arrange_kwargs(kwargs, defaults, order)
    badargs = setdiff(map(kw -> kw.args[1], kwargs), order)
    @assert isempty(badargs) "unrecognized keyword arguments: $(badargs)"
    return [:(Val{$(getkw(kwargs, kwsym, defaults))}()) for kwsym in order]
end

function getkw(kwargs, kwsym, defaults)
    for kwexpr in kwargs
        if kwexpr.args[1] == kwsym
            return kwexpr.args[2]
        end
    end
    return default_value(defaults, kwsym)
end

function default_value(defaults, kwsym)
    for kwpair in defaults
        if kwpair.first == kwsym
            return kwpair.second
        end
    end
    throw(KeyError(kwsym))
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

@eval cachefetch!{D,L}(::Type{D}, ::Type{L}) = $(Expr(:tuple, [:(cachefetch!($i, D, L)) for i in 1:NTHREADS]...))

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

function fetchxdiff{N,L}(x, chunk::Val{N}, xlength::Val{L})
    return cachefetch!(compat_threadid(), DiffNumber{N,eltype(x)}, xlength)
end

function fetchseeds{N,T}(::Vector{DiffNumber{N,T}}, args...)
    return cachefetch!(compat_threadid(), Partials{N,T}, args...)
end

#######################
# seeding work arrays #
#######################

function seedall!{N,T,L}(xdiff::Vector{DiffNumber{N,T}}, x, xlength::Val{L}, seed::Partials{N,T})
    @simd for i in 1:L
        @inbounds xdiff[i] = DiffNumber{N,T}(x[i], seed)
    end
    return xdiff
end

function seed!{N,T}(xdiff::Vector{DiffNumber{N,T}}, x, seed::Partials{N,T}, offset)
    k = offset - 1
    @simd for i in 1:N
        j = i + k
        @inbounds xdiff[j] = DiffNumber{N,T}(x[j], seed)
    end
    return xdiff
end

function seed!{N,T}(xdiff::Vector{DiffNumber{N,T}}, x, seeds::Vector{Partials{N,T}}, offset)
    k = offset - 1
    @simd for i in 1:N
        j = i + k
        @inbounds xdiff[j] = DiffNumber{N,T}(x[j], seeds[i])
    end
    return xdiff
end
