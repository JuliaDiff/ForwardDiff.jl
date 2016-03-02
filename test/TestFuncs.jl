module TestFuncs


#######################
# f: Vector -> Number #
#######################

@noinline function rosenbrock(x::AbstractVector)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

@noinline function ackley(x::AbstractVector)
    a, b, c = 20.0, -0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += i^2
    end
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + e)
end

@noinline self_weighted_logit(x::AbstractVector) = inv(1.0 + exp(-dot(x, x)))

const VECTOR_TO_NUMBER_FUNCS = (rosenbrock, ackley, self_weighted_logit)

#######################
# f: Vector -> Vector #
#######################

# Credit for adding `chebyquad!`, `brown_almost_linear!`, and `trigonometric!` goes to
# Kristoffer Carlsson (@KristofferC) - I (@jrevels) just ported them over from PR #104.

# Taken from NLsolve.jl
function chebyquad!(out::Vector, x::Vector)
    n = length(x)
    tk = 1/n
    for j = 1:n
        temp1 = 1.0
        temp2 = 2x[j]-1
        temp = 2temp2
        for i = 1:n
            out[i] += temp2
            ti = temp*temp2 - temp1
            temp1 = temp2
            temp2 = ti
        end
    end
    iev = -1.0
    for k = 1:n
        out[k] *= tk
        if iev > 0
            out[k] += 1/(k^2-1)
        end
        iev = -iev
    end
end

chebyquad(x) = (out = similar(x); chebyquad!(out, x); return out)

 # Taken from NLsolve.jl
 function brown_almost_linear!(x::Vector, fvec::Vector)
     n = length(x)
     sum1 = sum(x) - (n+1)
     for k = 1:(n-1)
         fvec[k] = x[k] + sum1
     end
     fvec[n] = prod(x) - 1
 end


 # Taken from NLsolve.jl
 function trigonometric!(x::Vector, fvec::Vector)
     n = length(x)
     for j = 1:n
         fvec[j] = cos(x[j])
     end
     sum1 = sum(fvec)
     for k = 1:n
         fvec[k] = n+k-sin(x[k]) - sum1 - k*fvec[k]
     end
 end

const VECTOR_TO_VECTOR_INPLACE_FUNCS = (chebyquad!, brown_almost_linear!, and trigonometric!)
const VECTOR_TO_VECTOR_FUNCS = (chebyquad, brown_almost_linear, and trigonometric)

end # module
