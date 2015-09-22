######################
# Taking Derivatives #
######################

# Exposed API methods #
#---------------------#
@generated function derivative!{A}(output, f, x::Number, ::Type{A}=Void)
    if A <: Void
        return_stmt = :(derivative!(output, result))
    elseif A <: AllResults
        return_stmt = :(derivative!(output, result), result)
    else
        error("invalid argument $A passed to FowardDiff.derivative")
    end

    return quote
        result = ForwardDiffResult(f(GradientNumber(x, one(x))))
        $return_stmt
    end
end

@generated function derivative{A}(f, x::Number, ::Type{A}=Void)

    if A <: Void
        return_stmt = :(derivative(result))
    elseif A <: AllResults
        return_stmt = :(derivative(result), result)
    else
        error("invalid argument $A passed to FowardDiff.derivative")
    end

    return quote
        result = ForwardDiffResult(f(GradientNumber(x, one(x))))
        $return_stmt
    end
end

function derivative{A}(f, ::Type{A}=Void; mutates=false)
    if mutates
        d!(output, x::Number) = ForwardDiff.derivative!(output, f, x, A)
        return d!
    else
        d(x::Number) = ForwardDiff.derivative(f, x, A)
        return d
    end
end
