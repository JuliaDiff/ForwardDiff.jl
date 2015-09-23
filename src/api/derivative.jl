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

# The below code generation enables better type inferencing in the event that
# `f` is a type (see https://github.com/JuliaDiff/ForwardDiff.jl/issues/54).
closure_deriv_def = quote
    if mutates
        d!(output, x::Number) = ForwardDiff.derivative!(output, f, x, A)
        return d!
    else
        d(x::Number) = ForwardDiff.derivative(f, x, A)
        return d
    end
end

@eval begin
    derivative{A}(f, ::Type{A}=Void; mutates=false) = $closure_deriv_def
    derivative{A,f}(::Type{f}, ::Type{A}=Void; mutates=false) = $closure_deriv_def
end
