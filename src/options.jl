abstract AbstractConfig

###########
# Config #
###########

immutable Config{N,T,D} <: AbstractConfig
    seeds::NTuple{N,Partials{N,T}}
    duals::D
    # disable default outer constructor
    function Config(seeds, duals)
        @assert N <= MAX_CHUNK_SIZE "cannot create Config{$N}: max chunk size is $(MAX_CHUNK_SIZE)"
        new(seeds, duals)
    end
end

# This is type-unstable, which is why our docs advise users to manually enter a chunk size
# when possible. The type instability here doesn't really hurt performance, since most of
# the heavy lifting happens behind a function barrier, but it can cause inference to give up
# when predicting the final output type of API functions.
Config(x::AbstractArray) = Config{pickchunksize(length(x))}(x)
Config(y::AbstractArray, x::AbstractArray) = Config{pickchunksize(length(x))}(y, x)

@compat (::Type{Config{N}}){N,T}(x::AbstractArray{T}) = begin
    seeds = construct_seeds(Partials{N,T})
    duals = similar(x, Dual{N,T})
    return Config{N,T,typeof(duals)}(seeds, duals)
end

@compat (::Type{Config{N}}){N,Y,X}(y::AbstractArray{Y}, x::AbstractArray{X}) = begin
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{N,Y})
    xduals = similar(x, Dual{N,X})
    duals = (yduals, xduals)
    return Config{N,X,typeof(duals)}(seeds, duals)
end

Base.copy{N,T,D}(cfg::Config{N,T,D}) = Config{N,T,D}(cfg.seeds, copy(cfg.duals))
Base.copy{N,T,D<:Tuple}(cfg::Config{N,T,D}) = Config{N,T,D}(cfg.seeds, map(copy, cfg.duals))

chunksize{N}(::Config{N}) = N
chunksize{N}(::Tuple{Vararg{Config{N}}}) = N

##################
# HessianConfig #
##################

immutable HessianConfig{N,J,JD,G,GD} <: AbstractConfig
    gradient_options::Config{N,G,GD}
    jacobian_options::Config{N,J,JD}
end

HessianConfig(x::AbstractArray) = HessianConfig{pickchunksize(length(x))}(x)
HessianConfig(out, x::AbstractArray) = HessianConfig{pickchunksize(length(x))}(out, x)

@compat (::Type{HessianConfig{N}}){N}(x::AbstractArray) = begin
    jacobian_options = Config{N}(x)
    gradient_options = Config{N}(jacobian_options.duals)
    return HessianConfig(gradient_options, jacobian_options)
end

@compat (::Type{HessianConfig{N}}){N}(out::DiffResult, x::AbstractArray) = begin
    jacobian_options = Config{N}(DiffBase.gradient(out), x)
    yduals, xduals = jacobian_options.duals
    gradient_options = Config{N}(xduals)
    return HessianConfig(gradient_options, jacobian_options)
end

Base.copy(cfg::HessianConfig) = HessianConfig(copy(cfg.gradient_options),
                                                 copy(cfg.jacobian_options))

@inline chunksize{N}(::HessianConfig{N}) = N
@inline chunksize{N}(::Tuple{Vararg{HessianConfig{N}}}) = N

gradient_options(cfg::HessianConfig) = cfg.gradient_options
jacobian_options(cfg::HessianConfig) = cfg.jacobian_options

###############
# Multithread #
###############

immutable Multithread{A,B} <: AbstractConfig
    options1::A
    options2::B
end

@eval function Multithread(cfg::Config)
    options1 = ntuple(n -> copy(cfg), Val{$NTHREADS})
    return Multithread(options1, nothing)
end

function Multithread(cfg::HessianConfig)
    options1 = Multithread(gradient_options(cfg))
    options2 = copy(jacobian_options(cfg))
    return Multithread(options1, options2)
end

gradient_options(cfg::Multithread) = cfg.options1
jacobian_options(cfg::Multithread) = cfg.options2

@inline chunksize(cfg::Multithread) = chunksize(gradient_options(cfg))
