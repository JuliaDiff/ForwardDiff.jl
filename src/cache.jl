const CACHE = ntuple(n -> Dict{DataType,Any}(), NTHREADS)

function clearcache!()
    for d in CACHE
        empty!(d)
    end
end

@eval cachefetch!(D::DataType, L::DataType) = $(Expr(:tuple, [:(cachefetch!($i, D, L)) for i in 1:NTHREADS]...))

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

function cachefetch!{N,T,Z}(tid::Integer, ::Type{Partials{N,T}}, ::Type{Val{Z}})
    K = Tuple{Partials{N,T},Z}
    V = Vector{Partials{N,T}}
    cache = CACHE[tid]
    if haskey(cache, K)
        seed_partials = cache[K]::V
    else
        seed_partials = V(Z)
        x = one(T)
        for i in 1:Z
            seed_partials[i] = setindex(zero(Partials{N,T}), x, i)
        end
        cache[K] = seed_partials
    end
    return seed_partials::V
end

cachefetch!{N,T}(tid::Integer, ::Type{Partials{N,T}}) = cachefetch!(tid, Partials{N,T}, Val{N})
