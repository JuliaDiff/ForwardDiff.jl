##########################
# picking the chunk size #
##########################

# Constrained to chunk <= MAX_CHUNK_SIZE, minimize (in order of priority):
#   1. the number of chunks that need to be computed
#   2. the number of "left over" perturbations in the final chunk
function pickchunksize(k)
    if k <= MAX_CHUNK_SIZE
        return k
    else
        nchunks = round(Int, k / MAX_CHUNK_SIZE, RoundUp)
        return round(Int, k / nchunks, RoundUp)
    end
end

####################
# value extraction #
####################

@inline extract_value!(out::DiffResult, ydual) = DiffBase.value!(value, out, ydual)
@inline extract_value!(out, ydual) = nothing

@inline function extract_value!(out, y, ydual)
    map!(value, y, ydual)
    copy_value!(out, y)
end

@inline copy_value!(out::DiffResult, y) = DiffBase.value!(out, y)
@inline copy_value!(out, y) = nothing

###################################
# vector mode function evaluation #
###################################

vector_mode_dual_eval(f, x, opts::Multithread) = vector_mode_dual_eval(f, x, gradient_options(opts))
vector_mode_dual_eval(f, x, opts::Tuple) = vector_mode_dual_eval(f, x, first(opts))

function vector_mode_dual_eval(f, x, opts)
    xdual = opts.duals
    seed!(xdual, x, opts.seeds)
    return f(xdual)
end

function vector_mode_dual_eval(f!, y, x, opts)
    ydual, xdual = opts.duals
    seed!(xdual, x, opts.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
    return ydual
end

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
