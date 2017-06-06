###############
# API methods #
###############

const AllowedDerivativeConfig{F,H} = Union{DerivativeConfig{Tag{F,H}}, DerivativeConfig{Tag{Void,H}}}

derivative(f!, y, x, cfg::DerivativeConfig) = throw(ConfigHismatchError(f, cfg))
derivative!(out, f!, y, x, cfg::DerivativeConfig) = throw(ConfigHismatchError(f, cfg))

@inline function derivative(f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, R))
    return extract_derivative(f(Dual{T}(x, one(x))))
end

@inline function derivative(f!::F, y, x::R, cfg::AllowedDerivativeConfig{F,H} = DerivativeConfig(f!, y, x)) where {F,R<:Real,H}
    ydual = cfg.duals
    seed!(ydual, y)
    f!(ydual, Dual{Tag{F,H}}(x, one(x)))
    map!(value, y, ydual)
    return extract_derivative(ydual)
end

@inline function derivative!(out, f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, R))
    ydual = f(Dual{T}(x, one(x)))
    out = extract_value!(out, ydual)
    out = extract_derivative!(out, ydual)
    return out
end

@inline function derivative!(out, f!::F, y, x::R, cfg::AllowedDerivativeConfig{F,H} = DerivativeConfig(f!, y, x)) where {F,R<:Real,H}
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

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)
extract_derivative!(out::DiffResult, y) = DiffBase.derivative!(extract_derivative, out, y)
