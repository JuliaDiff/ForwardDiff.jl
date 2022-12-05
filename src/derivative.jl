###############
# API methods #
###############

"""
    ForwardDiff.derivative(f, x::Real)

Return `df/dx` evaluated at `x`, assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Union{Real,AbstractArray})`.
"""
@inline function derivative(f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(f, R))
    return extract_derivative(T, f(Dual{T}(x, one(x))))
end

"""
    ForwardDiff.derivative(f!, y::AbstractArray, x::Real, cfg::DerivativeConfig = DerivativeConfig(f!, y, x), check=Val{true}())

Return `df!/dx` evaluated at `x`, assuming `f!` is called as `f!(y, x)` where the result is
stored in `y`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
@inline function derivative(f!, y::AbstractArray, x::Real,
                            cfg::DerivativeConfig{T} = DerivativeConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {T, CHK}
    CHK && checktag(T, f!, x)
    ydual = cfg.duals
    seed!(ydual, y)
    f!(ydual, Dual{T}(x, one(x)))
    map!(value, y, ydual)
    return extract_derivative(T, ydual)
end

"""
    ForwardDiff.derivative!(result::Union{AbstractArray,DiffResult}, f, x::Real)

Compute `df/dx` evaluated at `x` and store the result(s) in `result`, assuming `f` is called
as `f(x)`.

This method assumes that `isa(f(x), Union{Real,AbstractArray})`.
"""
@inline function derivative!(result::Union{AbstractArray,DiffResult},
                             f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(f, R))
    ydual = f(Dual{T}(x, one(x)))
    result = extract_value!(T, result, ydual)
    result = extract_derivative!(T, result, ydual)
    return result
end

"""
    ForwardDiff.derivative!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::Real, cfg::DerivativeConfig = DerivativeConfig(f!, y, x), check=Val{true}())

Compute `df!/dx` evaluated at `x` and store the result(s) in `result`, assuming `f!` is
called as `f!(y, x)` where the result is stored in `y`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
@inline function derivative!(result::Union{AbstractArray,DiffResult},
                             f!, y::AbstractArray, x::Real,
                             cfg::DerivativeConfig{T} = DerivativeConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {T, CHK}
    CHK && checktag(T, f!, x)
    ydual = cfg.duals
    seed!(ydual, y)
    f!(ydual, Dual{T}(x, one(x)))
    result = extract_value!(T, result, y, ydual)
    result = extract_derivative!(T, result, ydual)
    return result
end

derivative(f, x::AbstractArray) = throw(DimensionMismatch("derivative(f, x) expects that x is a real number. Perhaps you meant gradient(f, x)?"))
derivative(f, x::Complex) = throw(DimensionMismatch("derivative(f, x) expects that x is a real number (does not support Wirtinger derivatives). Separate real and imaginary parts of the input."))

#####################
# result extraction #
#####################

# non-mutating #
#--------------#

@inline extract_derivative(::Type{T}, y::Real) where {T}          = zero(y)
@inline extract_derivative(::Type{T}, y::Complex) where {T}       = zero(y)
@inline extract_derivative(::Type{T}, y::Dual) where {T}          = partials(T, y, 1)
@inline extract_derivative(::Type{T}, y::AbstractArray) where {T} = map(d -> extract_derivative(T,d), y)
@inline function extract_derivative(::Type{T}, y::Complex{TD}) where {T, TD <: Dual}
    complex(partials(T, real(y), 1), partials(T, imag(y), 1))
end

# mutating #
#----------#

extract_derivative!(::Type{T}, result::AbstractArray, y::AbstractArray) where {T} =
    map!(d -> extract_derivative(T,d), result, y)
extract_derivative!(::Type{T}, result::DiffResult, y) where {T} =
    DiffResults.derivative!(d -> extract_derivative(T,d), result, y)
