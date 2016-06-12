import ForwardDiff

using Base.Test

const XLEN = ForwardDiff.MAX_CHUNK_SIZE
const YLEN = div(ForwardDiff.MAX_CHUNK_SIZE, 2) + 1
const X, Y = rand(XLEN), rand(YLEN)
const CHUNK_SIZES = (1, ForwardDiff.pickchunksize(X), div(XLEN, 2) + 1, XLEN)
const EPS = 1e-5

# used to test against results calculated via finite difference
test_approx_eps(a::Array, b::Array) = @test_approx_eq_eps a b EPS

##################
# Test Functions #
##################

# f: Vector -> Number #
#---------------------#

function rosenbrock(x::AbstractVector)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

function ackley(x::AbstractVector)
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

self_weighted_logit(x::AbstractVector) = inv(1.0 + exp(-dot(x, x)))

const VECTOR_TO_NUMBER_FUNCS = (rosenbrock, ackley, self_weighted_logit)

# f: Vector -> Vector #
#---------------------#

# Credit for `chebyquad!`, `brown_almost_linear!`, and `trigonometric!` goes to
# Kristoffer Carlsson (@KristofferC) - I (@jrevels) just ported them over from PR #104.
# I've modified these functions to allow for tunable input/output sizes. These changes
# might make these functions incorrect in terms of their original design, but shouldn't
# be too different computationally (which is what we care about for tests/benchmarks).

function chebyquad!(y::Vector, x::Vector)
    tk = 1/length(x)
    for j = 1:length(x)
        temp1 = 1.0
        temp2 = 2x[j]-1
        temp = 2temp2
        for i = 1:length(y)
            y[i] += temp2
            ti = temp*temp2 - temp1
            temp1 = temp2
            temp2 = ti
        end
    end
    iev = -1.0
    for k = 1:length(y)
        y[k] *= tk
        if iev > 0
            y[k] += 1/(k^2-1)
        end
        iev = -iev
    end
    return y
end

chebyquad(x) = (y = zeros(eltype(x), length(Y)); chebyquad!(y, x); return y)

function brown_almost_linear!(y::Vector, x::Vector)
    c = sum(x) - (length(x) + 1)
    for i = 1:(length(x)-1), j = 1:(length(y)-1)
        y[j] += x[i] + c
    end
    y[length(y)] = prod(x) - 1
    return nothing
end

brown_almost_linear(x) = (y = zeros(eltype(x), length(Y)); brown_almost_linear!(y, x); return y)

function trigonometric!(y::Vector, x::Vector)
    for i in x, j in eachindex(y)
        y[j] = cos(i)
    end
    c = sum(y)
    n = length(x)
    for i in x, j in eachindex(y)
        y[j] = sin(i) * y[j] + n - c
    end
    return y
end

trigonometric(x) = (y = zeros(eltype(x), length(Y)); trigonometric!(y, x); return y)

const VECTOR_TO_VECTOR_INPLACE_FUNCS = (chebyquad!, brown_almost_linear!, trigonometric!)
const VECTOR_TO_VECTOR_FUNCS = (chebyquad, brown_almost_linear, trigonometric)
