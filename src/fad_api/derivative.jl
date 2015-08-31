######################
# Taking Derivatives #
######################

# Exposed API methods #
#---------------------#
derivative!(output, f, x::Number) = get_derivative!(output, f(GradientNumber(x, one(x))))
derivative(f, x::Number) = get_derivative(f(GradientNumber(x, one(x))))

function derivative(f; mutates=false)
    if mutates
        derivf!(output, x::Number) = ForwardDiff.derivative!(output, f, x)
        return derivf!
    else
        derivf(x::Number) = ForwardDiff.derivative(f, x)
        return derivf
    end
end
