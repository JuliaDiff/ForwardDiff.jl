##################
# chunk handling #
##################

immutable Chunk{N}
    function Chunk()
        @assert N <= (MAX_CHUNK_SIZE + 1) "cannot create Chunk{$N}: max chunk size is $(MAX_CHUNK_SIZE)"
        return new()
    end
end

@inline Base.copy(chunk::Chunk) = chunk

const AUTO_CHUNK_THRESHOLD = 10

pickchunk(x) = Chunk{pickchunksize(x)}()

function pickchunksize(x)
    k = length(x)
    if k <= AUTO_CHUNK_THRESHOLD
        return k
    else
        # Constrained to chunk <= AUTO_CHUNK_THRESHOLD, minimize (in order of priority):
        #   1. the number of chunks that need to be computed
        #   2. the number of "left over" perturbations in the final chunk
        nchunks = round(Int, k / AUTO_CHUNK_THRESHOLD, RoundUp)
        return round(Int, k / nchunks, RoundUp)
    end
end

#######################
# caching work arrays #
#######################

const CACHE = ntuple(n -> Dict(), NTHREADS)

function clearcache!()
    for d in CACHE
        empty!(d)
    end
end

@eval cachefetch!{D}(::Type{D}, n) = $(Expr(:tuple, [:(cachefetch!($i, D, n)) for i in 1:NTHREADS]...))

function cachefetch!{T}(tid::Integer, ::Type{T}, n, alt::Bool = false)
    K = (T, n, alt)
    V = Vector{T}
    cache = CACHE[tid]
    if haskey(cache, K)
        v = cache[K]::V
    else
        v = V(n)
        cache[K] = v
    end
    return v::V
end

function cachefetch!{N,T,Z}(tid::Integer, ::Type{Partials{N,T}}, ::Chunk{Z})
    K = (Partials{N,T}, Z)
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
    return cachefetch!(tid, Partials{N,T}, Chunk{N}())
end

function threaded_fetchdualvec{N}(x, ::Chunk{N})
    return cachefetch!(Dual{N,eltype(x)}, length(x))
end

function fetchdualvec{N}(x, ::Chunk{N}, alt::Bool = false)
    return cachefetch!(compat_threadid(), Dual{N,eltype(x)}, length(x), alt)
end

function fetchdualvechess{N}(x, ::Chunk{N})
    return cachefetch!(compat_threadid(), Dual{N,Dual{N,eltype(x)}}, length(x))
end

function fetchseeds{N,T}(::Type{Dual{N,T}}, args...)
    return cachefetch!(compat_threadid(), Partials{N,T}, args...)
end

#######################
# seeding work arrays #
#######################

# gradient/Jacobian versions

function seedall!{N,T}(xdual::Vector{Dual{N,T}}, x, seed::Partials{N,T})
    for i in eachindex(xdual)
        xdual[i] = Dual{N,T}(x[i], seed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,T}}, x, seed::Partials{N,T}, index)
    offset = index - 1
    for i in 1:N
        j = i + offset
        xdual[j] = Dual{N,T}(x[j], seed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,T}}, x, seeds::Vector{Partials{N,T}}, index,
                    chunksize = N)
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        xdual[j] = Dual{N,T}(x[j], seeds[i])
    end
    return xdual
end

# Hessian versions

function seedall!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x, inseed::Partials{N,T},
                       outseed::Partials{N,Dual{N,T}})
    for i in eachindex(xdual)
        xdual[i] = Dual{N,Dual{N,T}}(Dual{N,T}(x[i], inseed), outseed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,T}}, x, inseed::Partials{N,T},
                    outseed::Partials{N,Dual{N,T}}, index, chunksize = N)
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        xdual[j] = Dual{N,Dual{N,T}}(Dual{N,T}(x[j], inseed), outseed)
    end
    return xdual
end

function seed!{N,T}(xdual::Vector{Dual{N,T}}, x, inseeds::Vector{Partials{N,T}},
                    outseeds::Vector{Partials{N,Dual{N,T}}}, index, chunksize = N)
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        xdual[j] = Dual{N,Dual{N,T}}(Dual{N,T}(x[j], inseeds[i]), outseeds[i])
    end
    return xdual
end

sideseed!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, args...) = seed!(xdual, args..., N - 1)

function sideseedj!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x, inseed::Partials{N,T},
                         outseed::Partials{N,Dual{N,T}}, j)
    xdual[j] = Dual{N,Dual{N,T}}(Dual{N,T}(x[j], inseed), outseed)
    return xdual
end
