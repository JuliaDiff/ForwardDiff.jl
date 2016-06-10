####################
# DerivativeResult #
####################

type DerivativeResult{V,D} <: ForwardDiffResult
    value::V
    derivative::D
end

value(result::DerivativeResult) = result.value

derivative(result::DerivativeResult) = result.derivative

derivative!(out, result::DerivativeResult) = copy!(out, result.derivative)

###############
# API methods #
###############

derivative(f, x) = extract_derivative(f(Dual(x, one(x))))

function derivative!(out, f, x)
    dual = f(Dual(x, one(x)))
    load_derivative_value!(out, dual)
    load_derivative!(out, dual)
    return out
end

#####################
# result extraction #
#####################

@inline extract_derivative(dual::Dual) = partials(dual, 1)
@inline extract_derivative(arr) = load_derivative!(similar(arr, numtype(eltype(arr))), arr)

function load_derivative!(out, arr)
    for i in eachindex(out)
        out[i] = extract_derivative(arr[i])
    end
    return out
end

@inline function load_derivative!(out::DerivativeResult, arr)
    load_derivative!(out.derivative, arr)
    return out
end

@inline function load_derivative!(out::DerivativeResult, dual::Dual)
    out.derivative = extract_derivative(dual)
    return out
end

@inline load_derivative_value!(out, result) = out

function load_derivative_value!(out::DerivativeResult, arr)
    val = out.value
    for i in eachindex(val)
        val[i] = value(arr[i])
    end
    return out
end

@inline function load_derivative_value!(out::DerivativeResult, dual::Dual)
    out.value = value(dual)
    return out
end
