###############
# API methods #
###############

@inline function derivative(f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, R))
    return extract_derivative(f(Dual{T}(x, one(x))))
end

@inline function derivative!(out, f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, typeof(x)))
    extract_derivative!(out, f(Dual{T}(x, one(x))))
    return out
end

#####################
# result extraction #
#####################

# non-mutating #
#--------------#

@inline extract_derivative(y::Dual{T,V,1}) where {T,V} = partials(y, 1)
@inline extract_derivative(y::Real) = zero(y)
@inline extract_derivative(y::AbstractArray) = extract_derivative!(similar(y, valtype(eltype(y))), y)

# mutating #
#----------#

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)

function extract_derivative!(out::DiffResult, y)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(extract_derivative, out, y)
    return out
end
