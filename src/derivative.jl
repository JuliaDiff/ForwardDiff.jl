###############
# API methods #
###############

derivative{F}(f::F, x) = extract_derivative(f(Dual(x, one(x))))
derivative{F}(f::F, x, args...) = extract_derivative(f(Dual(x, one(x)), args...))

function derivative!{F}(out, f::F, x)
    y = f(Dual(x, one(x)))
    extract_derivative!(out, y)
    return out
end

#####################
# result extraction #
#####################

@inline extract_derivative(y::Real) = partials(y, 1)
@inline extract_derivative(y::AbstractArray) = extract_derivative!(similar(y, valtype(eltype(y))), y)

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)

function extract_derivative!(out::DiffResult, y)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(extract_derivative, out, y)
    return out
end
