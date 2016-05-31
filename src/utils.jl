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

@noinline function pickchunk{C,L}(chunk::Val{C}, len::Val{L}, x)
    @assert C <= MAX_CHUNK_SIZE "max chunk size is $(MAX_CHUNK_SIZE)"
    return (chunk, len)
end

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
    return v::V
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

function threaded_fetchxdual{L,N}(x, len::Val{L}, chunk::Val{N})
    return cachefetch!(Dual{N,eltype(x)}, len)
end

function fetchxdual{L,N}(x, len::Val{L}, chunk::Val{N})
    return cachefetch!(compat_threadid(), Dual{N,eltype(x)}, len)
end

function fetchxdualhess{L,N}(x, len::Val{L}, chunk::Val{N})
    return cachefetch!(compat_threadid(), Dual{N,Dual{N,eltype(x)}}, len)
end

function fetchseeds{N,T}(::Type{Dual{N,T}}, args...)
    return cachefetch!(compat_threadid(), Partials{N,T}, args...)
end

#######################
# seeding work arrays #
#######################

# gradient/Jacobian versions

function seedall!{N,T,L}(xdual::Vector{Dual{N,T}}, x, len::Val{L}, seed::Partials{N,T})
    @simd for i in 1:L
        @inbounds xdual[i] = Dual{N,T}(x[i], seed)
    end
    return xdual
end


function seed!{N,T}(xdual::Vector{Dual{N,T}}, x, offset, seed::Partials{N,T})
    k = offset - 1
    @simd for i in 1:N
        j = i + k
        @inbounds xdual[j] = Dual{N,T}(x[j], seed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,T}}, x, offset, seeds::Vector{Partials{N,T}})
    k = offset - 1
    @simd for i in 1:N
        j = i + k
        @inbounds xdual[j] = Dual{N,T}(x[j], seeds[i])
    end
    return xdual
end

# Hessian versions

function seedall!{N,T,L}(xdual::Vector{Dual{N,Dual{N,T}}}, x, len::Val{L},
                         inseed::Partials{N,T}, outseed::Partials{N,Dual{N,T}})
    @simd for i in 1:L
        @inbounds xdual[i] = Dual{N,Dual{N,T}}(Dual{N,T}(x[i], inseed), outseed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x, offset,
                    inseed::Partials{N,T}, outseed::Partials{N,Dual{N,T}},
                    M = N)
    k = offset - 1
    @simd for i in 1:M
        j = i + k
        @inbounds xdual[j] = Dual{N,Dual{N,T}}(Dual{N,T}(x[j], inseed), outseed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x, offset,
                    inseeds::Vector{Partials{N,T}}, outseeds::Vector{Partials{N,Dual{N,T}}},
                    M = N)
    k = offset - 1
    @simd for i in 1:M
        j = i + k
        @inbounds xdual[j] = Dual{N,Dual{N,T}}(Dual{N,T}(x[j], inseeds[i]), outseeds[i])
    end
    return xdual
end

offseed!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, args...) = seed!(xdual, args..., N - 1)

function offseedj!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x, j,
                        inseed::Partials{N,T}, outseed::Partials{N,Dual{N,T}})
    @inbounds xdual[j] = Dual{N,Dual{N,T}}(Dual{N,T}(x[j], inseed), outseed)
    return xdual
end
