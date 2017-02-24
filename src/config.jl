@compat abstract type AbstractConfig end

@inline chunksize(::Tuple{}) = error("empty tuple passed to `chunksize`")

##################
# GradientConfig #
##################

@compat immutable GradientConfig{N,V,D} <: AbstractConfig
    seeds::NTuple{N,Partials{N,V}}
    duals::D
    # disable default outer constructor
    (::Type{GradientConfig{N,V,D}}){N,V,D}(seeds, duals) = new{N,V,D}(seeds, duals)
end

GradientConfig(x::AbstractArray) = GradientConfig{pickchunksize(length(x))}(x)

function (::Type{GradientConfig{N}}){N,V}(x::AbstractArray{V})
    seeds = construct_seeds(Partials{N,V})
    duals = similar(x, Dual{N,V})
    return GradientConfig{N,V,typeof(duals)}(seeds, duals)
end

Base.copy{N,V,D}(cfg::GradientConfig{N,V,D}) = GradientConfig{N,V,D}(cfg.seeds, copy(cfg.duals))
Base.copy{N,V,D<:Tuple}(cfg::GradientConfig{N,V,D}) = GradientConfig{N,V,D}(cfg.seeds, map(copy, cfg.duals))

@inline chunksize{N}(::GradientConfig{N}) = N
@inline chunksize{N}(::Tuple{Vararg{GradientConfig{N}}}) = N

##################
# JacobianConfig #
##################

@compat immutable JacobianConfig{N,V,D} <: AbstractConfig
    seeds::NTuple{N,Partials{N,V}}
    duals::D
    # disable default outer constructor
    (::Type{JacobianConfig{N,V,D}}){N,V,D}(seeds, duals) = new{N,V,D}(seeds, duals)
end

JacobianConfig(x::AbstractArray) = JacobianConfig{pickchunksize(length(x))}(x)

function (::Type{JacobianConfig{N}}){N,V}(x::AbstractArray{V})
    seeds = construct_seeds(Partials{N,V})
    duals = similar(x, Dual{N,V})
    return JacobianConfig{N,V,typeof(duals)}(seeds, duals)
end

JacobianConfig(y::AbstractArray, x::AbstractArray) = JacobianConfig{pickchunksize(length(x))}(y, x)

function (::Type{JacobianConfig{N}}){N,Y,X}(y::AbstractArray{Y}, x::AbstractArray{X})
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{N,Y})
    xduals = similar(x, Dual{N,X})
    duals = (yduals, xduals)
    return JacobianConfig{N,X,typeof(duals)}(seeds, duals)
end

Base.copy{N,T,D}(cfg::JacobianConfig{N,T,D}) = JacobianConfig{N,T,D}(cfg.seeds, copy(cfg.duals))
Base.copy{N,T,D<:Tuple}(cfg::JacobianConfig{N,T,D}) = JacobianConfig{N,T,D}(cfg.seeds, map(copy, cfg.duals))

@inline chunksize{N}(::JacobianConfig{N}) = N
@inline chunksize{N}(::Tuple{Vararg{JacobianConfig{N}}}) = N

#################
# HessianConfig #
#################

immutable HessianConfig{N,J,JD,G,GD} <: AbstractConfig
    gradient_config::GradientConfig{N,G,GD}
    jacobian_config::JacobianConfig{N,J,JD}
end

HessianConfig(x::AbstractArray) = HessianConfig{pickchunksize(length(x))}(x)
HessianConfig(out, x::AbstractArray) = HessianConfig{pickchunksize(length(x))}(out, x)

function (::Type{HessianConfig{N}}){N}(x::AbstractArray)
    jacobian_config = JacobianConfig{N}(x)
    gradient_config = GradientConfig{N}(jacobian_config.duals)
    return HessianConfig(gradient_config, jacobian_config)
end

function (::Type{HessianConfig{N}}){N}(out::DiffResult, x::AbstractArray)
    jacobian_config = JacobianConfig{N}(DiffBase.gradient(out), x)
    yduals, xduals = jacobian_config.duals
    gradient_config = GradientConfig{N}(xduals)
    return HessianConfig(gradient_config, jacobian_config)
end

Base.copy(cfg::HessianConfig) = HessianConfig(copy(cfg.gradient_config),
                                              copy(cfg.jacobian_config))

@inline chunksize{N}(::HessianConfig{N}) = N
@inline chunksize{N}(::Tuple{Vararg{HessianConfig{N}}}) = N

gradient_config(cfg::HessianConfig) = cfg.gradient_config
jacobian_config(cfg::HessianConfig) = cfg.jacobian_config

#####################
# MultithreadConfig #
#####################

immutable MultithreadConfig{A,B} <: AbstractConfig
    config1::A
    config2::B
end

@eval function MultithreadConfig(cfg::Union{GradientConfig,JacobianConfig})
    config1 = ntuple(n -> copy(cfg), Val{$NTHREADS})
    return MultithreadConfig(config1, nothing)
end

function MultithreadConfig(cfg::HessianConfig)
    config1 = MultithreadConfig(gradient_config(cfg))
    config2 = copy(jacobian_config(cfg))
    return MultithreadConfig(config1, config2)
end

gradient_config(cfg::MultithreadConfig) = cfg.config1
jacobian_config(cfg::MultithreadConfig) = cfg.config2

@inline chunksize(cfg::MultithreadConfig) = chunksize(gradient_config(cfg))
