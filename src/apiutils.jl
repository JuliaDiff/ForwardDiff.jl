####################
# value extraction #
####################

@inline extract_value!(::Type{T}, out::DiffResult, ydual) where {T} =
    DiffResults.value!(d -> value(T,d), out, ydual)
@inline extract_value!(::Type{T}, out, ydual) where {T} = out # ???

@inline function extract_value!(::Type{T}, out, y, ydual) where {T}
    map!(d -> value(T,d), y, ydual)
    copy_value!(out, y)
end

@inline copy_value!(out::DiffResult, y) = DiffResults.value!(out, y)
@inline copy_value!(out, y) = out

###################################
# vector mode function evaluation #
###################################

@generated function dualize(::Type{T}, x::SArray{S,V,D,N}) where {T,S,V,D,N}
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    return quote
        chunk = Chunk{N}()
        $(Expr(:meta, :inline))
        return SArray{S}($(dx))
    end
end

@inline static_dual_eval(::Type{T}, f, x::SArray) where {T} = f(dualize(T, x))

function vector_mode_dual_eval(f::F, x, cfg::Union{JacobianConfig,GradientConfig}) where {F}
    xdual = cfg.duals
    seed!(xdual, x, cfg.seeds)
    return f(xdual)
end

function vector_mode_dual_eval(f!::F, y, x, cfg::JacobianConfig) where {F}
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
    duals .= Dual{T,V,N}.(x, Base.RefValue(seed))
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x,
               seeds::NTuple{N,Partials{N,V}}) where {T,V,N}
    duals[1:N] .= Dual{T,V,N}.(x[1:N], seeds[1:N])
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seed::Partials{N,V} = zero(Partials{N,V})) where {T,V,N}
    offset = index - 1
    chunk = (1:N) .+ offset
    duals[chunk] .= Dual{T,V,N}.(x[chunk], Base.RefValue(seed))
    return duals
end

function seed!(duals::AbstractArray{Dual{T,V,N}}, x, index,
               seeds::NTuple{N,Partials{N,V}}, chunksize = N) where {T,V,N}
    offset = index - 1
    chunk = (1:chunksize) .+ offset
    duals[chunk] .= Dual{T,V,N}.(x[chunk], seeds[1:chunksize])
    return duals
end
