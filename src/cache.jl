#######################################
# caching for Jacobians and gradients #
#######################################

const JACOBIAN_CACHE = Dict{Tuple{Int,Int,DataType,Bool},Any}()
const MULTITHREAD_JACOBIAN_CACHE = Dict{Tuple{Int,Int,DataType,Bool},Any}()

immutable JacobianCache{N,T}
    dualvec::Vector{Dual{N,T}}
    seeds::Vector{Partials{N,T}}
    remainder_seeds::Vector{Partials{N,T}}
end

function JacobianCache{T,N}(::Type{T}, xlen, chunk::Chunk{N})
    dualvec = Vector{Dual{N,T}}(xlen)
    seeds = construct_seeds(T, chunk)
    remainder = xlen % N
    if remainder == 0
        remainder_seeds = seeds
    else
        remainder_seeds = construct_seeds(T, chunk, remainder)
    end
    return JacobianCache{N,T}(dualvec, seeds, remainder_seeds)
end

function jacobian_cachefetch!{T,N}(::Type{T}, xlen, chunk::Chunk{N}, alt::Bool = false)
    key = (xlen, N, T, alt)
    if haskey(JACOBIAN_CACHE, key)
        return JACOBIAN_CACHE[key]::JacobianCache{N,T}
    else
        result = JacobianCache(T, xlen, chunk)
        JACOBIAN_CACHE[key] = result
        return result::JacobianCache{N,T}
    end
end

function jacobian_cachefetch!(x, chunk::Chunk, alt::Bool = false)
    return jacobian_cachefetch!(eltype(x), length(x), chunk, alt)
end

function multithread_jacobian_cachefetch!{T,N}(::Type{T}, xlen, ::Chunk{N}, alt::Bool = false)
    key = (xlen, N, T, alt)
    if haskey(MULTITHREAD_JACOBIAN_CACHE, key)
        return MULTITHREAD_JACOBIAN_CACHE[key]::Vector{JacobianCache{N,T}}
    else
        resultvec = Vector{JacobianCache{N,T}}(NTHREADS)
        result = JacobianCache(T, xlen, chunk, alt)
        resultvec[1] = result
        for i in 2:NTHREADS
            resultvec[i] = copy(result)
        end
        MULTITHREAD_JACOBIAN_CACHE[key] = resultvec
        return resultvec::Vector{JacobianCache{N,T}}
    end
end

function multithread_jacobian_cachefetch!(x, chunk::Chunk, alt::Bool = false)
    return multithread_jacobian_cachefetch!(eltype(x), length(x), chunk, alt)
end

########################
# caching for Hessians #
########################
# only used for vector mode, so we can assume that N == length(x)
const HESSIAN_CACHE = Dict{Tuple{Int,DataType},Any}()

immutable HessianCache{N,T}
    dualvec::Vector{Dual{N,Dual{N,T}}}
    inseeds::Vector{Partials{N,T}}
    outseeds::Vector{Partials{N,Dual{N,T}}}
end

function HessianCache{T,N}(::Type{T}, chunk::Chunk{N})
    dualvec = Vector{Dual{N,Dual{N,T}}}(N)
    inseeds = construct_seeds(T, chunk)
    outseeds = construct_seeds(Dual{N,T}, chunk)
    return HessianCache{N,T}(dualvec, inseeds, outseeds)
end

function hessian_cachefetch!{T,N}(::Type{T}, chunk::Chunk{N})
    key = (N, T)
    if haskey(HESSIAN_CACHE, key)
        return HESSIAN_CACHE[key]::HessianCache{N,T}
    else
        result = HessianCache(T, chunk)
        HESSIAN_CACHE[key] = result
        return result::HessianCache{N,T}
    end
end

function hessian_cachefetch!(x, chunk::Chunk)
    return hessian_cachefetch!(eltype(x), chunk)
end

#################
# Partial seeds #
#################

function construct_seeds{T,N}(::Type{T}, ::Chunk{N}, len = N)
    seeds = Vector{Partials{N,T}}(len)
    seed_unit = one(T)
    zero_partials = zero(Partials{N,T})
    for i in eachindex(seeds)
        seeds[i] = setindex(zero_partials, seed_unit, i)
    end
    return seeds
end

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

function seedhess!{N,T}(xdual::Vector{Dual{N,Dual{N,T}}}, x, inseeds::Vector{Partials{N,T}},
                        outseeds::Vector{Partials{N,Dual{N,T}}})
    for i in 1:N
        xdual[i] = Dual{N,Dual{N,T}}(Dual{N,T}(x[i], inseeds[i]), outseeds[i])
    end
    return xdual
end
