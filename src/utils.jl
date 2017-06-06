####################
# value extraction #
####################

@inline extract_value!(out::DiffResult, ydual) = DiffBase.value!(value, out, ydual)
@inline extract_value!(out, ydual) = out

@inline function extract_value!(out, y, ydual)
    map!(value, y, ydual)
    copy_value!(out, y)
end

@inline copy_value!(out::DiffResult, y) = DiffBase.value!(out, y)
@inline copy_value!(out, y) = out

###################################
# vector mode function evaluation #
###################################

@generated function dualize(::F, x::SArray{S,V,D,N}) where {F,S,V,D,N}
    tag = Tag(F, x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    return quote
        chunk = Chunk{N}()
        T = typeof($tag)
        $(Expr(:meta, :inline))
        return SArray{S}($(dx))
    end
end

@inline vector_mode_dual_eval(f::F, x::SArray) where {F} = f(dualize(f, x))

function vector_mode_dual_eval(f::F, x, cfg::Union{JacobianConfig,GradientConfig}) where F
    xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval(f!::F, y, x, cfg::JacobianConfig) where F
    ydual, xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
    return ydual
end

##################################
# seed construction/manipulation #
##################################

@generated function construct_seeds(::Type{Partials{N,V}}) where {N,V}
    return Expr(:tuple, [:(single_seed(Partials{N,V}, Val{$i}())) for i in 1:N]...)
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    for i in eachindex(duals)
        duals[i] = Dual{T,V,N}(x[i], seed)
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    for i in 1:N
        duals[i] = Dual{T,V,N}(x[i], seeds[i])
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    offset = index - 1
    for i in 1:N
        j = i + offset
        duals[j] = Dual{T,V,N}(x[j], seed)
    end
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    for i in 1:chunksize
        j = i + offset
        duals[j] = Dual{T,V,N}(x[j], seeds[i])
    end
    return duals
end
