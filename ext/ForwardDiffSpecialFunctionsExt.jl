module ForwardDiffSpecialFunctionsExt

using ForwardDiff: ForwardDiff, Dual, DiffRules, NaNMath, value, partials,
                   uses_specialfunctions, unary_dual_definition, binary_dual_definition
import SpecialFunctions

for (M, f, arity) in DiffRules.diffrules(filter_modules = nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        continue
    elseif !uses_specialfunctions(M, f, arity)
        continue
    end
    if arity == 1
        eval(unary_dual_definition(M, f))
    elseif arity == 2
        eval(binary_dual_definition(M, f))
    end
end

# Functions in SpecialFunctions which return tuples #
# Their derivatives are not defined in DiffRules    #
#---------------------------------------------------#

function SpecialFunctions.logabsgamma(d::Dual{T,<:Real}) where {T}
    x = value(d)
    y, s = SpecialFunctions.logabsgamma(x)
    return (Dual{T}(y, SpecialFunctions.digamma(x) * partials(d)), s)
end

# Derivatives wrt to first parameter and precision setting are not supported
function SpecialFunctions.gamma_inc(a::Real, d::Dual{T,<:Real}, ind::Integer) where {T}
    x = value(d)
    p, q = SpecialFunctions.gamma_inc(a, x, ind)
    ∂p = exp(-x) * x^(a - 1) / SpecialFunctions.gamma(a) * partials(d)
    return (Dual{T}(p, ∂p), Dual{T}(q, -∂p))
end

end # module
