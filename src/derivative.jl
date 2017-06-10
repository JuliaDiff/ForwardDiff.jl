###############
# API methods #
###############

const AllowedDerivativeConfig{F,H} = Union{DerivativeConfig{Tag{F,H}}, DerivativeConfig{Tag{Void,H}}}

derivative(f!, y, x, cfg::DerivativeConfig) = throw(ConfigHismatchError(f, cfg))
derivative!(out, f!, y, x, cfg::DerivativeConfig) = throw(ConfigHismatchError(f, cfg))

"""
    ForwardDiff.derivative(f, x::Real)

Return `df/dx` evaluated at `x`, assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Union{Real,AbstractArray})`.
"""
@inline function derivative(f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, R))
    return extract_derivative(f(Dual{T}(x, one(x))))
end

"""
    ForwardDiff.derivative(f!, y::AbstractArray, x::Real, cfg::DerivativeConfig = DerivativeConfig(f!, y, x))

Return `df!/dx` evaluated at `x`, assuming `f!` is called as `f!(y, x)` where the result is
stored in `y`.
"""
@inline function derivative(f!::F, y::AbstractArray, x::R,
                            cfg::AllowedDerivativeConfig{F,H} = DerivativeConfig(f!, y, x)) where {F,R<:Real,H}
    ydual = cfg.duals
    seed!(ydual, y)
    f!(ydual, Dual{Tag{F,H}}(x, one(x)))
    map!(value, y, ydual)
    return extract_derivative(ydual)
end

"""
    ForwardDiff.derivative!(result::Union{AbstractArray,DiffResult}, f, x::Real)

Compute `df/dx` evaluated at `x` and store the result(s) in `result`, assuming `f` is called
as `f(x)`.

This method assumes that `isa(f(x), Union{Real,AbstractArray})`.
"""
@inline function derivative!(result::Union{AbstractArray,DiffResult},
                             f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, R))
    ydual = f(Dual{T}(x, one(x)))
    out = extract_value!(out, ydual)
    out = extract_derivative!(out, ydual)
    return out
end

"""
    ForwardDiff.derivative!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::Real, cfg::DerivativeConfig = DerivativeConfig(f!, y, x))

Compute `df!/dx` evaluated at `x` and store the result(s) in `result`, assuming `f!` is
called as `f!(y, x)` where the result is stored in `y`.
"""
@inline function derivative!(result::Union{AbstractArray,DiffResult},
                             f!::F, y::AbstractArray, x::R,
                             cfg::AllowedDerivativeConfig{F,H} = DerivativeConfig(f!, y, x)) where {F,R<:Real,H}
    ydual = cfg.duals
    seed!(ydual, y)
    f!(ydual, Dual{Tag{F,H}}(x, one(x)))
    out = extract_value!(out, y, ydual)
    out = extract_derivative!(out, ydual)
    return out
end

#####################
# result extraction #
#####################

# non-mutating #
#--------------#

@inline extract_derivative(y::Dual{T,V,1}) where {T,V} = partials(y, 1)
@inline extract_derivative(y::Real) = zero(y)
@inline extract_derivative(y::AbstractArray) = map(extract_derivative, y)

# mutating #
#----------#

extract_derivative!(result::AbstractArray, y::AbstractArray) = map!(extract_derivative, result, y)
extract_derivative!(result::DiffResult, y) = DiffBase.derivative!(extract_derivative, result, y)
