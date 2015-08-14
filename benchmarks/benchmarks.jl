using ForwardDiff

##################
# Test functions #
##################
sqr(i) = i^2
twopicos(i) = cos(2*π*i)

function ackley(x)
    len_recip = 1/length(x)
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += twopicos(i)
        sum_sqrs += sqr(i)
    end
    return -20 * exp(-0.2 * sqrt(len_recip*sum_sqrs)) - exp(len_recip * sum_cos) + 20 + e
end

#############################
# Benchmark utility methods #
#############################
# Usage:
#
# # The values of the Dict are Arrays of time values where indices correspond to length(x)
# julia> t = bench_fad(ackley, 10:10:100, 4) # benchmark ackley where length(x) = 10:10:100, taking the minimum of 4 trials
# Dict{Symbol,Array{Float64,1}} with 3 entries:
#   :gtimes => [4.152e-6,1.8057e-5,2.0612e-5,3.546e-5,3.9163e-5,9.3705e-5,5.6879e-5,6.3378e-5,9.8352e-5,0.000106597]
#   :htimes => [2.5515e-5,7.4746e-5,0.000164618,0.000345889,0.000773864,0.00110334,0.002396697,0.003085223,0.004590107,…
#   :ftimes => [7.83e-7,9.26e-7,1.11e-6,1.162e-6,1.324e-6,1.504e-6,1.569e-6,1.766e-6,2.002e-6,2.056e-6]
#
# julia> t = bench_fad(ackley, 400, 4)
# Dict{Symbol,Array{Float64,1}} with 3 entries:
#   :gtimes => [0.00109457]
#   :htimes => [0.725081113]
#   :ftimes => [6.836e-6]

function bench_fad(f, range, repeat=3)
    g = ForwardDiff.gradient(f)
    h = ForwardDiff.hessian(f)

    # warm up
    bench_range(f, range, 1)
    bench_range(g, range, 1)
    bench_range(h, range, 1)

    # actual
    return Dict(
        :ftimes => bench_range(f,range,repeat),
        :gtimes => bench_range(g,range,repeat),
        :htimes => bench_range(h,range,repeat)
    )
end

function bench_func(f, xlen, repeat)
    x=rand(xlen)
    min_time = Inf
    for i in 1:repeat
        this_time = (tic(); f(x); toq())
        min_time = min(this_time, min_time)
    end
    return min_time
end

function bench_range(f, range, repeat=3)
    return [bench_func(f, xlen, repeat) for xlen in range]
end

