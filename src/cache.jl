####################################
# AbstractArray eltype replacement #
####################################
function eltype_param_number(T)
    if T.name.name == :AbstractArray
        return 1
    else
        T_super = supertype(T)
        param_number = eltype_param_number(T_super)
        tv = T_super.parameters[param_number]
        for i = 1:T.parameters.length
            if tv == T.parameters[i]
                return i
            end
        end
    end
end

@inline Base.@pure @generated function replace_eltype{T,S}(x::AbstractArray{T}, ::Type{S})
    pnum = eltype_param_number(x.name.primary)
    tname = x.name.name
    tparams = collect(x.parameters)
    tparams[pnum] = S
    newtype = :($(tname){$(tparams...)})
    return newtype
end

#######################################
# caching for Jacobians and gradients #
#######################################

const JACOBIAN_CACHE = Dict{Tuple{Int,Int,DataType,Bool},Any}()

immutable JacobianCache{N,T,D}
    duals::D
    seeds::NTuple{N,Partials{N,T}}
end

function JacobianCache{N}(x, chunk::Chunk{N})
    T = eltype(x)
    duals = similar(x, Dual{N,T}, size(x))
    seeds = construct_seeds(T, chunk)
    return JacobianCache{N,T,typeof(duals)}(duals, seeds)
end

@inline jacobian_dual_type{T,M,N}(arr::AbstractArray{T,M}, ::Chunk{N}) = replace_eltype(arr, Dual{N,T})

Base.copy(cache::JacobianCache) = JacobianCache(copy(cache.duals), cache.seeds)

@eval function multithread_jacobian_cachefetch!{N}(x, chunk::Chunk{N}, usecache::Bool,
                                                   alt::Bool = false)
    T, xlen = eltype(x), length(x)
    if usecache
        result = get!(JACOBIAN_CACHE, (xlen, N, T, alt)) do
            construct_jacobian_caches(x, chunk)
        end
    else
        result = construct_jacobian_caches(x, chunk)
    end
    return result::NTuple{$NTHREADS,JacobianCache{N,T,jacobian_dual_type(x, chunk)}}
end

jacobian_cachefetch!(args...) = multithread_jacobian_cachefetch!(args...)[compat_threadid()]

########################
# caching for Hessians #
########################

# only used for vector mode, so we can assume that N == length(x)
const HESSIAN_CACHE = Dict{Tuple{Int,DataType},Any}()

immutable HessianCache{N,T,D}
    duals::D
    inseeds::NTuple{N,Partials{N,T}}
    outseeds::NTuple{N,Partials{N,Dual{N,T}}}
end

function HessianCache{N}(x, chunk::Chunk{N})
    T = eltype(x)
    duals = similar(x, Dual{N,Dual{N,T}}, size(x))
    inseeds = construct_seeds(T, chunk)
    outseeds = construct_seeds(Dual{N,T}, chunk)
    return HessianCache{N,T,typeof(duals)}(duals, inseeds, outseeds)
end

@inline hessian_dual_type{T,M,N}(arr::AbstractArray{T,M}, ::Chunk{N}) = replace_eltype(arr, Dual{N,Dual{N,T}})

Base.copy(cache::HessianCache) = HessianCache(copy(cache.duals), cache.inseeds, cache.outseeds)

@eval function multithread_hessian_cachefetch!{N}(x, chunk::Chunk{N}, usecache::Bool)
    T = eltype(x)
    if usecache
        result = get!(HESSIAN_CACHE, (N, T)) do
            construct_hessian_caches(x, chunk)
        end
    else
        result = construct_hessian_caches(x, chunk)
    end
    return result::NTuple{$NTHREADS,HessianCache{N,T,hessian_dual_type(x, chunk)}}
end

hessian_cachefetch!(args...) = multithread_hessian_cachefetch!(args...)[compat_threadid()]

#################
# Partial seeds #
#################

function seedall!{N,T}(xdual, x, seed::Partials{N,T})
    for i in eachindex(xdual)
        xdual[i] = Dual{N,T}(x[i], seed)
    end
    return xdual
end

function seed!{N,T}(xdual, x, seed::Partials{N,T}, index)
    offset = index - 1
    for i in 1:N
        j = i + offset
        xdual[j] = Dual{N,T}(x[j], seed)
    end
    return xdual
end

function seed!{N,T}(xdual, x,seeds::NTuple{N,Partials{N,T}}, index, chunksize = N)
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        xdual[j] = Dual{N,T}(x[j], seeds[i])
    end
    return xdual
end

function seedhess!{N,T}(xdual, x, inseeds::NTuple{N,Partials{N,T}},
                        outseeds::NTuple{N,Partials{N,Dual{N,T}}})
    for i in 1:N
        xdual[i] = Dual{N,Dual{N,T}}(Dual{N,T}(x[i], inseeds[i]), outseeds[i])
    end
    return xdual
end

#####################
# @eval'd functions #
#####################

@eval function construct_jacobian_caches{N}(x, chunk::Chunk{N})
    result = JacobianCache(x, chunk)
    return $(Expr(:tuple, :result, [:(copy(result)) for i in 2:NTHREADS]...))
end

@eval function construct_hessian_caches{N}(x, chunk::Chunk{N})
    result = HessianCache(x, chunk)
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
