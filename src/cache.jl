const CACHE = ntuple(n -> Dict{DataType,Any}(), Threads.nthreads())

function clearcache!()
    for d in CACHE
        empty!(d)
    end
end

@eval cachefetch!(D::DataType, L::DataType) = $(Expr(:tuple, [:(cachefetch!($i, D, L)) for i in 1:Threads.nthreads()]...))

function cachefetch!{N,T,L}(tid::Integer, ::Type{DiffNumber{N,T}}, ::Type{Val{L}})
    K = Tuple{DiffNumber{N,T},L}
    V = Vector{DiffNumber{N,T}}
    d = CACHE[tid]
    if haskey(d, K)
        workvec = d[K]::V
    else
        workvec = V(L)
        d[K] = workvec
    end
    return workvec::V
end

function cachefetch!{N,T,Z}(tid::Integer, ::Type{Partials{N,T}}, ::Type{Val{Z}})
    K = Tuple{Partials{N,T},Z}
    V = Vector{Partials{N,T}}
    d = CACHE[tid]
    if haskey(d, K)
        seed_partials = d[K]::V
    else
        seed_partials = V(Z)
        x = one(T)
        for i in 1:Z
            seed_partials[i] = setindex(zero(Partials{N,T}), x, i)
        end
        d[K] = seed_partials
    end
    return seed_partials::V
end

cachefetch!{N,T}(tid::Integer, ::Type{Partials{N,T}}) = cachefetch!(tid, Partials{N,T}, Val{N})
