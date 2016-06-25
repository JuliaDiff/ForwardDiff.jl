#######################################
# caching for Jacobians and gradients #
#######################################

const JACOBIAN_CACHE = Dict{Tuple{Int,Int,DataType,Bool},Any}()

immutable JacobianCache{N,T}
    dualvec::Vector{Dual{N,T}}
    seeds::NTuple{N,Partials{N,T}}
end

function JacobianCache{T,N}(::Type{T}, xlen, chunk::Chunk{N})
    dualvec = Vector{Dual{N,T}}(xlen)
    seeds = construct_seeds(T, chunk)
    return JacobianCache{N,T}(dualvec, seeds)
end

Base.copy(cache::JacobianCache) = JacobianCache(copy(cache.dualvec), cache.seeds)

@eval function multithread_jacobian_cachefetch!{T,N}(::Type{T}, xlen, chunk::Chunk{N},
                                                     usecache::Bool, alt::Bool = false)
    if usecache
        key = (xlen, N, T, alt)
        if haskey(JACOBIAN_CACHE, key)
            return JACOBIAN_CACHE[key]::NTuple{$NTHREADS,JacobianCache{N,T}}
        else
            allresults = construct_jacobian_caches(T, xlen, chunk)
            JACOBIAN_CACHE[key] = allresults
            return allresults::NTuple{$NTHREADS,JacobianCache{N,T}}
        end
    else
        return construct_jacobian_caches(T, xlen, chunk)::NTuple{$NTHREADS,JacobianCache{N,T}}
    end
end

multithread_jacobian_cachefetch!(x, args...) = multithread_jacobian_cachefetch!(eltype(x), length(x), args...)

jacobian_cachefetch!(args...) = multithread_jacobian_cachefetch!(args...)[compat_threadid()]

########################
# caching for Hessians #
########################

# only used for vector mode, so we can assume that N == length(x)
const HESSIAN_CACHE = Dict{Tuple{Int,DataType},Any}()

immutable HessianCache{N,T}
    dualvec::Vector{Dual{N,Dual{N,T}}}
    inseeds::NTuple{N,Partials{N,T}}
    outseeds::NTuple{N,Partials{N,Dual{N,T}}}
end

function HessianCache{T,N}(::Type{T}, chunk::Chunk{N})
    dualvec = Vector{Dual{N,Dual{N,T}}}(N)
    inseeds = construct_seeds(T, chunk)
    outseeds = construct_seeds(Dual{N,T}, chunk)
    return HessianCache{N,T}(dualvec, inseeds, outseeds)
end

Base.copy(cache::HessianCache) = HessianCache(copy(cache.dualvec), cache.inseeds, cache.outseeds)

@eval function multithread_hessian_cachefetch!{T,N}(::Type{T}, chunk::Chunk{N}, usecache::Bool)
    if usecache
        key = (N, T)
        if haskey(HESSIAN_CACHE, key)
            return HESSIAN_CACHE[key]::NTuple{$NTHREADS,HessianCache{N,T}}
        else
            allresults = construct_hessian_caches(T, chunk)
            HESSIAN_CACHE[key] = allresults
            return allresults::NTuple{$NTHREADS,HessianCache{N,T}}
        end
    else
        return construct_hessian_caches(T, chunk)::NTuple{$NTHREADS,HessianCache{N,T}}
    end
end

multithread_hessian_cachefetch!(x, args...) = multithread_hessian_cachefetch!(eltype(x), args...)

hessian_cachefetch!(args...) = multithread_hessian_cachefetch!(args...)[compat_threadid()]

#################
# Partial seeds #
#################

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

function seed!{N,T}(xdual::Vector{Dual{N,T}}, x,seeds::NTuple{N,Partials{N,T}}, index, chunksize = N)
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        xdual[j] = Dual{N,T}(x[j], seeds[i])
    end
    return xdual
end

function seedhess!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x,
                        inseeds::NTuple{N,Partials{N,T}},
                        outseeds::NTuple{N,Partials{N,Dual{N,T}}})
    for i in 1:N
        xdual[i] = Dual{N,Dual{N,T}}(Dual{N,T}(x[i], inseeds[i]), outseeds[i])
    end
    return xdual
end

#####################
# @eval'd functions #
#####################

@eval function construct_jacobian_caches{T,N}(::Type{T}, xlen, chunk::Chunk{N})
    result = JacobianCache(T, xlen, chunk)
    return $(Expr(:tuple, :result, [:(copy(result)) for i in 2:NTHREADS]...))
end

@eval function construct_hessian_caches{T,N}(::Type{T}, chunk::Chunk{N})
    result = HessianCache(T, chunk)
    return $(Expr(:tuple, :result, [:(copy(result)) for i in 2:NTHREADS]...))
end

for N in 1:MAX_CHUNK_SIZE
    ex = Expr(:tuple, [:(setindex(zero_partials, seed_unit, $i)) for i in 1:N]...)
    @eval function construct_seeds{T}(::Type{T}, ::Chunk{$N})
        seed_unit = one(T)
        zero_partials = zero(Partials{$N,T})
        return $ex
    end
end
