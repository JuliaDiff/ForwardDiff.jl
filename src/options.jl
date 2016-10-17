###########
# Options #
###########

immutable Options{N,T,D}
    seeds::NTuple{N,Partials{N,T}}
    duals::D
    # disable default outer constructor
    function Options(seeds, duals)
        @assert N <= MAX_CHUNK_SIZE "cannot create Options{$N}: max chunk size is $(MAX_CHUNK_SIZE)"
        new(seeds, duals)
    end
end

# This is type-unstable, which is why our docs advise users to manually enter a chunk size
# when possible. The type instability here doesn't really hurt performance, since most of
# the heavy lifting happens behind a function barrier, but it can cause inference to give up
# when predicting the final output type of API functions.
Options(x::AbstractArray) = Options{pickchunksize(length(x))}(x)
Options(y::AbstractArray, x::AbstractArray) = Options{pickchunksize(length(x))}(y, x)
Options{N,T,D<:AbstractArray}(opts::Options{N,T,D}) = Options(opts.duals)

function Options{N,T,D<:Tuple}(opts::Options{N,T,D})
    ydual, xdual = opts.duals
    return Options(xdual)
end

function (::Type{Options{N}}){N,T}(x::AbstractArray{T})
    seeds = construct_seeds(Partials{N,T})
    duals = similar(x, Dual{N,T})
    return Options{N,T,typeof(duals)}(seeds, duals)
end

function (::Type{Options{N}}){N,Y,X}(y::AbstractArray{Y}, x::AbstractArray{X})
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{N,Y})
    xduals = similar(x, Dual{N,X})
    duals = (yduals, xduals)
    return Options{N,X,typeof(duals)}(seeds, duals)
end

Base.copy{N,T,D}(opts::Options{N,T,D}) = Options{N,T,D}(opts.seeds, copy(opts.duals))

chunksize{N}(::Options{N}) = N
chunksize{N}(::Tuple{Vararg{Options{N}}}) = N

##################################
# seed construction/manipulation #
##################################

for N in 1:MAX_CHUNK_SIZE
    ex = Expr(:tuple, [:(setindex(zero_partials, seed_unit, $i)) for i in 1:N]...)
    @eval function construct_seeds{T}(::Type{Partials{$N,T}})
        seed_unit = one(T)
        zero_partials = zero(Partials{$N,T})
        return $ex
    end
end

function seed!{N,T}(duals::AbstractArray{Dual{N,T}}, x,
                    seed::Partials{N,T} = zero(Partials{N,T}))
    for i in eachindex(duals)
        duals[i] = Dual{N,T}(x[i], seed)
    end
    return duals
end

function seed!{N,T}(duals::AbstractArray{Dual{N,T}}, x,
                    seeds::NTuple{N,Partials{N,T}})
    for i in 1:N
        duals[i] = Dual{N,T}(x[i], seeds[i])
    end
    return duals
end

function seed!{N,T}(duals::AbstractArray{Dual{N,T}}, x, index,
                    seed::Partials{N,T} = zero(Partials{N,T}))
    offset = index - 1
    for i in 1:N
        j = i + offset
        duals[j] = Dual{N,T}(x[j], seed)
    end
    return duals
end

function seed!{N,T}(duals::AbstractArray{Dual{N,T}}, x, index,
                    seeds::NTuple{N,Partials{N,T}}, chunksize = N)
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        duals[j] = Dual{N,T}(x[j], seeds[i])
    end
    return duals
end

# function seedhess!{N,T}(duals, x, inseeds::NTuple{N,Partials{N,T}},
#                         outseeds::NTuple{N,Partials{N,Dual{N,T}}})
#     for i in 1:N
#         duals[i] = Dual{N,Dual{N,T}}(Dual{N,T}(x[i], inseeds[i]), outseeds[i])
#     end
#     return duals
# end
