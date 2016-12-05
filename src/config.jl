abstract AbstractConfig

###########
# Config #
###########

@inline chunksize(::Tuple{}) = error("empty tuple passed to `chunksize`")

# Define a few different AbstractConfig types. All these types share the same structure,
# but feature different constructors and dispatch restrictions in downstream code.
for Config in (:GradientConfig, :JacobianConfig)
    @eval begin
        immutable $Config{N,T,D} <: AbstractConfig
            seeds::NTuple{N,Partials{N,T}}
            duals::D
            # disable default outer constructor
            function $Config(seeds, duals)
                @assert N <= CHUNK_THRESHOLD
                new(seeds, duals)
            end
        end

        # This is type-unstable, which is why our docs advise users to manually enter a chunk size
        # when possible. The type instability here doesn't really hurt performance, since most of
        # the heavy lifting happens behind a function barrier, but it can cause inference to give up
        # when predicting the final output type of API functions.
        $Config(x::AbstractArray) = $Config{pickchunksize(length(x))}(x)

        @compat (::Type{$Config{N}}){N,T}(x::AbstractArray{T}) = begin
            seeds = construct_seeds(Partials{N,T})
            duals = similar(x, Dual{N,T})
            return $Config{N,T,typeof(duals)}(seeds, duals)
        end

        Base.copy{N,T,D}(cfg::$Config{N,T,D}) = $Config{N,T,D}(cfg.seeds, copy(cfg.duals))
        Base.copy{N,T,D<:Tuple}(cfg::$Config{N,T,D}) = $Config{N,T,D}(cfg.seeds, map(copy, cfg.duals))

        @inline chunksize{N}(::$Config{N}) = N
        @inline chunksize{N}(::Tuple{Vararg{$Config{N}}}) = N
    end
end

JacobianConfig(y::AbstractArray, x::AbstractArray) = JacobianConfig{pickchunksize(length(x))}(y, x)

@compat (::Type{JacobianConfig{N}}){N,Y,X}(y::AbstractArray{Y}, x::AbstractArray{X}) = begin
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{N,Y})
    xduals = similar(x, Dual{N,X})
    duals = (yduals, xduals)
    return JacobianConfig{N,X,typeof(duals)}(seeds, duals)
end

##################
# HessianConfig #
##################

immutable HessianConfig{N,J,JD,G,GD} <: AbstractConfig
    gradient_config::GradientConfig{N,G,GD}
    jacobian_config::JacobianConfig{N,J,JD}
end

HessianConfig(x::AbstractArray) = HessianConfig{pickchunksize(length(x))}(x)
HessianConfig(out, x::AbstractArray) = HessianConfig{pickchunksize(length(x))}(out, x)

@compat (::Type{HessianConfig{N}}){N}(x::AbstractArray) = begin
    jacobian_config = JacobianConfig{N}(x)
    gradient_config = GradientConfig{N}(jacobian_config.duals)
    return HessianConfig(gradient_config, jacobian_config)
end

@compat (::Type{HessianConfig{N}}){N}(out::DiffResult, x::AbstractArray) = begin
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
