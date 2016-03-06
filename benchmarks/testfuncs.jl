######################################
# Test Functions f: Vector -> Number #
######################################
# Below functions must:
#
# - Take a single Vector argument `x` and return a Number
# - ...where `x` can be an arbitrary length
# - Be type stable for arbitrary `eltype(x)`
# - Be listed in `vec2num_testfuncs` below
# - Be exported

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

function ackley(x)
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

self_weighted_logit(x) = inv(1.0 + exp(-dot(x, x)))

const vec2num_testfuncs = (rosenbrock, ackley, self_weighted_logit)

export rosenbrock, ackley, self_weighted_logit


######################################
# Test Functions f: Number -> Number #
######################################
# Below functions must:
#
# - Take a single argument `x` and return a Number
# - ...where `x` can be an arbitrary Number
# - Be type stable for arbitrary `typeof(x)`

# Fourier series of a sawtooth wave
function sawtooth(x)
    N = 50
    s = zero(x)
    for n = 1:N
        s += (-1)^(n+1) * sin(n * x) / n
    end
    return 2/π * s
end

# Taylor series for sin(x)
function taylor_sin{T}(x::T)
    N = 50
    s = zero(x)
    for n = 0:N
        s += (-1)^n * x^(2n + 1) / factorial(T(2n + 1))
    end
    return s
end

