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

function vector_mode_dual_eval{F}(f::F, x, cfg::Union{JacobianConfig,GradientConfig})
    xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval{F}(f!::F, y, x, cfg::JacobianConfig)
    ydual, xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
    return ydual
end

##################################
# seed construction/manipulation #
##################################

@generated function construct_seeds{N,T}(::Type{Partials{N,T}})
    return Expr(:tuple, [:(single_seed(Partials{N,T}, Val{$i})) for i in 1:N]...)
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
