abstract AbstractOptions

###########
# Options #
###########

immutable Options{N,T,D} <: AbstractOptions
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

@compat (::Type{Options{N}}){N,T}(x::AbstractArray{T}) = begin
    seeds = construct_seeds(Partials{N,T})
    duals = similar(x, Dual{N,T})
    return Options{N,T,typeof(duals)}(seeds, duals)
end

@compat (::Type{Options{N}}){N,Y,X}(y::AbstractArray{Y}, x::AbstractArray{X}) = begin
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{N,Y})
    xduals = similar(x, Dual{N,X})
    duals = (yduals, xduals)
    return Options{N,X,typeof(duals)}(seeds, duals)
end

Base.copy{N,T,D}(opts::Options{N,T,D}) = Options{N,T,D}(opts.seeds, copy(opts.duals))
Base.copy{N,T,D<:Tuple}(opts::Options{N,T,D}) = Options{N,T,D}(opts.seeds, map(copy, opts.duals))

chunksize{N}(::Options{N}) = N
chunksize{N}(::Tuple{Vararg{Options{N}}}) = N

##################
# HessianOptions #
##################

immutable HessianOptions{N,J,JD,G,GD} <: AbstractOptions
    gradient_options::Options{N,G,GD}
    jacobian_options::Options{N,J,JD}
end

HessianOptions(x::AbstractArray) = HessianOptions{pickchunksize(length(x))}(x)
HessianOptions(out, x::AbstractArray) = HessianOptions{pickchunksize(length(x))}(out, x)

@compat (::Type{HessianOptions{N}}){N}(x::AbstractArray) = begin
    jacobian_options = Options{N}(x)
    gradient_options = Options{N}(jacobian_options.duals)
    return HessianOptions(gradient_options, jacobian_options)
end

@compat (::Type{HessianOptions{N}}){N}(out::DiffResult, x::AbstractArray) = begin
    jacobian_options = Options{N}(DiffBase.gradient(out), x)
    yduals, xduals = jacobian_options.duals
    gradient_options = Options{N}(xduals)
    return HessianOptions(gradient_options, jacobian_options)
end

Base.copy(opts::HessianOptions) = HessianOptions(copy(opts.gradient_options),
                                                 copy(opts.jacobian_options))

@inline chunksize{N}(::HessianOptions{N}) = N
@inline chunksize{N}(::Tuple{Vararg{HessianOptions{N}}}) = N

gradient_options(opts::HessianOptions) = opts.gradient_options
jacobian_options(opts::HessianOptions) = opts.jacobian_options

###############
# Multithread #
###############

immutable Multithread{A,B} <: AbstractOptions
    options1::A
    options2::B
end

@eval function Multithread(opts::Options)
    options1 = ntuple(n -> copy(opts), Val{$NTHREADS})
    return Multithread(options1, nothing)
end

function Multithread(opts::HessianOptions)
    options1 = Multithread(gradient_options(opts))
    options2 = copy(jacobian_options(opts))
    return Multithread(options1, options2)
end

gradient_options(opts::Multithread) = opts.options1
jacobian_options(opts::Multithread) = opts.options2

@inline chunksize(opts::Multithread) = chunksize(gradient_options(opts))
