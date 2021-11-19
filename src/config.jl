#######
# Tag #
#######

struct Tag{F,V}
end

const TAGCOUNT = Threads.Atomic{UInt}(0)

# each tag is assigned a unique number
# tags which depend on other tags will be larger
@generated function tagcount(::Type{Tag{F,V}}) where {F,V}
    :($(Threads.atomic_add!(TAGCOUNT, UInt(1))))
end

function Tag(f::F, ::Type{V}) where {F,V}
    tagcount(Tag{F,V}) # trigger generated function
    Tag{F,V}()
end

Tag(::Nothing, ::Type{V}) where {V} = nothing


@inline function ≺(::Type{Tag{F1,V1}}, ::Type{Tag{F2,V2}}) where {F1,V1,F2,V2}
    tagcount(Tag{F1,V1}) < tagcount(Tag{F2,V2})
end

struct InvalidTagException{E,O} <: Exception
end

Base.showerror(io::IO, e::InvalidTagException{E,O}) where {E,O} =
    print(io, "Invalid Tag object:\n  Expected $E,\n  Observed $O.")

checktag(::Type{Tag{FT,VT}}, f::F, x::AbstractArray{V}) where {FT,VT,F,V} =
    throw(InvalidTagException{Tag{F,V},Tag{FT,VT}}())

checktag(::Type{Tag{F,V}}, f::F, x::AbstractArray{V}) where {F,V} = true

# no easy way to check Jacobian tag used with Hessians as multiple functions may be used
checktag(::Type{Tag{FT,VT}}, f::F, x::AbstractArray{V}) where {FT<:Tuple,VT,F,V} = true

# custom tag: you're on your own.
checktag(z, f, x) = true


##################
# AbstractConfig #
##################

abstract type AbstractConfig{N} end

Base.copy(cfg::AbstractConfig) = deepcopy(cfg)

Base.eltype(cfg::AbstractConfig) = eltype(typeof(cfg))

@inline (chunksize(::AbstractConfig{N})::Int) where {N} = N

####################
# DerivativeConfig #
####################

struct DerivativeConfig{T,D} <: AbstractConfig{1}
    duals::D
end

"""
    ForwardDiff.DerivativeConfig(f!, y::AbstractArray, x::AbstractArray)

Return a `DerivativeConfig` instance based on the type of `f!`, and the types/shapes of the
output vector `y` and the input vector `x`.

The returned `DerivativeConfig` instance contains all the work buffers required by
`ForwardDiff.derivative` and `ForwardDiff.derivative!` when the target function takes the form
`f!(y, x)`.

If `f!` is `nothing` instead of the actual target function, then the returned instance can
be used with any target function. However, this will reduce ForwardDiff's ability to catch
and prevent perturbation confusion (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

This constructor does not store/modify `y` or `x`.
"""
function DerivativeConfig(f::F,
                          y::AbstractArray{Y},
                          x::X,
                          tag::T = Tag(f, X)) where {F,X<:Real,Y<:Real,T}
    duals = similar(y, Dual{T,Y,1})
    return DerivativeConfig{T,typeof(duals)}(duals)
end

checktag(::DerivativeConfig{T},f,x) where {T} = checktag(T,f,x)
Base.eltype(::Type{DerivativeConfig{T,D}}) where {T,D} = eltype(D)

##################
# GradientConfig #
##################

struct GradientConfig{T,V,N,D} <: AbstractConfig{N}
    seeds::NTuple{N,Partials{N,V}}
    duals::D
end

"""
    ForwardDiff.GradientConfig(f, x::AbstractArray, chunk::Chunk = Chunk(x))

Return a `GradientConfig` instance based on the type of `f` and type/shape of the input
vector `x`.

The returned `GradientConfig` instance contains all the work buffers required by
`ForwardDiff.gradient` and `ForwardDiff.gradient!`.

If `f` is `nothing` instead of the actual target function, then the returned instance can
be used with any target function. However, this will reduce ForwardDiff's ability to catch
and prevent perturbation confusion (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

This constructor does not store/modify `x`.
"""
function GradientConfig(f::F,
                        x::AbstractArray{V},
                        ::Chunk{N} = Chunk(x),
                        ::T = Tag(f, V)) where {F,V,N,T}
    seeds = construct_seeds(Partials{N,V})
    duals = similar(x, Dual{T,V,N})
    return GradientConfig{T,V,N,typeof(duals)}(seeds, duals)
end

checktag(::GradientConfig{T},f,x) where {T} = checktag(T,f,x)
Base.eltype(::Type{GradientConfig{T,V,N,D}}) where {T,V,N,D} = Dual{T,V,N}

##################
# JacobianConfig #
##################

struct JacobianConfig{T,V,N,D} <: AbstractConfig{N}
    seeds::NTuple{N,Partials{N,V}}
    duals::D
end

"""
    ForwardDiff.JacobianConfig(f, x::AbstractArray, chunk::Chunk = Chunk(x))

Return a `JacobianConfig` instance based on the type of `f` and type/shape of the input
vector `x`.

The returned `JacobianConfig` instance contains all the work buffers required by
`ForwardDiff.jacobian` and `ForwardDiff.jacobian!` when the target function takes the form
`f(x)`.

If `f` is `nothing` instead of the actual target function, then the returned instance can
be used with any target function. However, this will reduce ForwardDiff's ability to catch
and prevent perturbation confusion (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

This constructor does not store/modify `x`.
"""
function JacobianConfig(f::F,
                        x::AbstractArray{V},
                        ::Chunk{N} = Chunk(x),
                        ::T = Tag(f, V)) where {F,V,N,T}
    seeds = construct_seeds(Partials{N,V})
    duals = similar(x, Dual{T,V,N})
    return JacobianConfig{T,V,N,typeof(duals)}(seeds, duals)
end

"""
    ForwardDiff.JacobianConfig(f!, y::AbstractArray, x::AbstractArray, chunk::Chunk = Chunk(x))

Return a `JacobianConfig` instance based on the type of `f!`, and the types/shapes of the
output vector `y` and the input vector `x`.

The returned `JacobianConfig` instance contains all the work buffers required by
`ForwardDiff.jacobian` and `ForwardDiff.jacobian!` when the target function takes the form
`f!(y, x)`.

If `f!` is `nothing` instead of the actual target function, then the returned instance can
be used with any target function. However, this will reduce ForwardDiff's ability to catch
and prevent perturbation confusion (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

This constructor does not store/modify `y` or `x`.
"""
function JacobianConfig(f::F,
                        y::AbstractArray{Y},
                        x::AbstractArray{X},
                        ::Chunk{N} = Chunk(x),
                        ::T = Tag(f, X)) where {F,Y,X,N,T}
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{T,Y,N})
    xduals = similar(x, Dual{T,X,N})
    duals = (yduals, xduals)
    return JacobianConfig{T,X,N,typeof(duals)}(seeds, duals)
end

checktag(::JacobianConfig{T},f,x) where {T} = checktag(T,f,x)
Base.eltype(::Type{JacobianConfig{T,V,N,D}}) where {T,V,N,D} = Dual{T,V,N}

#################
# HessianConfig #
#################

struct HessianConfig{T,V,N,DG,DJ} <: AbstractConfig{N}
    jacobian_config::JacobianConfig{T,V,N,DJ}
    gradient_config::GradientConfig{T,Dual{T,V,N},N,DG}
end

"""
    ForwardDiff.HessianConfig(f, x::AbstractArray, chunk::Chunk = Chunk(x))

Return a `HessianConfig` instance based on the type of `f` and type/shape of the input
vector `x`.

The returned `HessianConfig` instance contains all the work buffers required by
`ForwardDiff.hessian` and `ForwardDiff.hessian!`. For the latter, the buffers are
configured for the case where the `result` argument is an `AbstractArray`. If
it is a `DiffResult`, the `HessianConfig` should instead be constructed via
`ForwardDiff.HessianConfig(f, result, x, chunk)`.

If `f` is `nothing` instead of the actual target function, then the returned instance can
be used with any target function. However, this will reduce ForwardDiff's ability to catch
and prevent perturbation confusion (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

This constructor does not store/modify `x`.
"""
function HessianConfig(f::F,
                       x::AbstractArray{V},
                       chunk::Chunk = Chunk(x),
                       tag = Tag(f, V)) where {F,V}
    jacobian_config = JacobianConfig(f, x, chunk, tag)
    gradient_config = GradientConfig(f, jacobian_config.duals, chunk, tag)
    return HessianConfig(jacobian_config, gradient_config)
end

"""
    ForwardDiff.HessianConfig(f, result::DiffResult, x::AbstractArray, chunk::Chunk = Chunk(x))

Return a `HessianConfig` instance based on the type of `f`, types/storage in `result`, and
type/shape of the input vector `x`.

The returned `HessianConfig` instance contains all the work buffers required by
`ForwardDiff.hessian!` for the case where the `result` argument is an `DiffResult`.

If `f` is `nothing` instead of the actual target function, then the returned instance can
be used with any target function. However, this will reduce ForwardDiff's ability to catch
and prevent perturbation confusion (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

This constructor does not store/modify `x`.
"""
function HessianConfig(f::F,
                       result::DiffResult,
                       x::AbstractArray{V},
                       chunk::Chunk = Chunk(x),
                       tag = Tag(f, V)) where {F,V}
    jacobian_config = JacobianConfig((f,gradient), DiffResults.gradient(result), x, chunk, tag)
    gradient_config = GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    return HessianConfig(jacobian_config, gradient_config)
end

checktag(::HessianConfig{T},f,x) where {T} = checktag(T,f,x)
Base.eltype(::Type{HessianConfig{T,V,N,DG,DJ}}) where {T,V,N,DG,DJ} =
    Dual{T,Dual{T,V,N},N}
