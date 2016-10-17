###############
# API methods #
###############

derivative(f, x) = extract_derivative(f(Dual(x, one(x))))

function derivative!(out, f, x)
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
