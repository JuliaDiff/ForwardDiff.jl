######################
# Taking Derivatives #
######################

# Exposed API methods #
#---------------------#
derivative!(output::Array, f, x::Number) = load_derivative!(output, f(GradientNumber(x, one(x))))
derivative(f, x::Number) = load_derivative(f(GradientNumber(x, one(x))))

function derivative(f; mutates=false)
    if mutates
        derivf!(output::Array, x::Number) = derivative!(output, f, x)
        return derivf!
    else
        derivf(x::Number) = derivative(f, x)
        return derivf
    end
end

# Helper functions #
#------------------#
function load_derivative!(output::Array, arr::Array)
    @assert length(arr) == length(output)
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(arr[i], 1)
    end
    return output
end

load_derivative(arr::Array) = load_derivative!(similar(arr, eltype(eltype(arr))), arr)
load_derivative(n::ForwardDiffNumber{1}) = grad(n, 1)
