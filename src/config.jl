#######
# Tag #
#######

struct Tag{F,H} end

const TAGCOUNT = Threads.Atomic{UInt}(0)

# each tag is assigned a unique number
# there is a potential for error if Duals are shared across processes
# e.g. by saving/loading or parallel computations
@generated function Tag(::Type{F}, ::Type{V}) where {F,V}
    n = Threads.atomic_add!(TAGCOUNT, UInt(1))
    return quote
        $(Expr(:meta, :inline))
        Tag{F,$n}()
    end
end

@inline ≺(::Type{Tag{F1,n1}}, ::Type{Tag{F2,n2}}) where {F1,n1,F2,n2} = n1 < n2

Base.@pure function Base.promote_rule(::Type{Dual{T1,V1,N1}},
                                      ::Type{Dual{T2,V2,N2}}) where {T1,V1<:Real,N1,T2,V2<:Real,N2}
    V = promote_type(V1,V2)
    if T2 ≺ T1
        Dual{T1,Dual{T2,V,N2},N1}
    else
        Dual{T2,Dual{T1,V,N1},N2}
    end
end


##################
# AbstractConfig #
##################

abstract type AbstractConfig{T,N} end

struct ConfigMismatchError{F,G,H} <: Exception
    f::F
    cfg::AbstractConfig{Tag{G,H}}
end

function Base.showerror(io::IO, e::ConfigMismatchError{F,G}) where {F,G}
    print(io, "The provided configuration (of type $(typeof(e.cfg))) was constructed for a",
              " function other than the current target function. ForwardDiff cannot safely",
              " perform differentiation in this context; see the following issue for details:",
              " https://github.com/JuliaDiff/ForwardDiff.jl/issues/83. You can resolve this",
              " problem by constructing and using a configuration with the appropriate target",
              " function, e.g. `ForwardDiff.GradientConfig($(e.f), x)`")
end

Base.copy(cfg::AbstractConfig) = deepcopy(cfg)

Base.eltype(cfg::AbstractConfig) = eltype(typeof(cfg))

@inline chunksize(::AbstractConfig{T,N}) where {T,N} = N

####################
# DerivativeConfig #
####################

struct DerivativeConfig{T,D} <: AbstractConfig{T,1}
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
function DerivativeConfig(::F,
                          y::AbstractArray{Y},
                          x::X,
                          ::T = Tag(F, X)) where {F,X<:Real,Y<:Real,T}
    duals = similar(y, Dual{T,Y,1})
    return DerivativeConfig{T,typeof(duals)}(duals)
end

Base.eltype(::Type{DerivativeConfig{T,D}}) where {T,D} = eltype(D)

##################
# GradientConfig #
##################

struct GradientConfig{T,V,N,D} <: AbstractConfig{T,N}
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
function GradientConfig(::F,
                        x::AbstractArray{V},
                        ::Chunk{N} = Chunk(x),
                        ::T = Tag(F, V)) where {F,V,N,T}
    seeds = construct_seeds(Partials{N,V})
    duals = similar(x, Dual{T,V,N})
    return GradientConfig{T,V,N,typeof(duals)}(seeds, duals)
end

Base.eltype(::Type{GradientConfig{T,V,N,D}}) where {T,V,N,D} = Dual{T,V,N}

##################
# JacobianConfig #
##################

struct JacobianConfig{T,V,N,D} <: AbstractConfig{T,N}
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
function JacobianConfig(::F,
                        x::AbstractArray{V},
                        ::Chunk{N} = Chunk(x),
                        ::T = Tag(F, V)) where {F,V,N,T}
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
function JacobianConfig(::F,
                        y::AbstractArray{Y},
                        x::AbstractArray{X},
                        ::Chunk{N} = Chunk(x),
                        ::T = Tag(F, X)) where {F,Y,X,N,T}
    seeds = construct_seeds(Partials{N,X})
    yduals = similar(y, Dual{T,Y,N})
    xduals = similar(x, Dual{T,X,N})
    duals = (yduals, xduals)
    return JacobianConfig{T,X,N,typeof(duals)}(seeds, duals)
end

Base.eltype(::Type{JacobianConfig{T,V,N,D}}) where {T,V,N,D} = Dual{T,V,N}

#################
# HessianConfig #
#################

struct HessianConfig{T,V,N,D,H,DJ} <: AbstractConfig{T,N}
    jacobian_config::JacobianConfig{Tag{Void,H},V,N,DJ}
    gradient_config::GradientConfig{T,Dual{Tag{Void,H},V,N},N,D}
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
                       tag::Tag = Tag(F, Dual{Void,V,0})) where {F,V}
    jacobian_config = JacobianConfig(nothing, x, chunk)
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
                       tag::Tag = Tag(F, Dual{Void,V,0})) where {F,V}
    jacobian_config = JacobianConfig(nothing, DiffResults.gradient(result), x, chunk)
    gradient_config = GradientConfig(f, jacobian_config.duals[2], chunk, tag)
    return HessianConfig(jacobian_config, gradient_config)
end

Base.eltype(::Type{HessianConfig{T,V,N,D,H,DJ}}) where {T,V,N,D,H,DJ} = Dual{T,Dual{Tag{Void,H},V,N},N}
