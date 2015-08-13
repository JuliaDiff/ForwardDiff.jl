using ForwardDiff

##################
# Test functions #
##################
ackley_term(a,b) = (-20 * exp(-0.2 * sqrt(0.5*(a^2 + b^2))) 
                   - exp(0.5 * (cos(2pi * a) + cos(2pi * b))) + e + 20)

function ackley_sum(x::Vector)
    result = zero(eltype(x))
    @simd for i in 1:length(x)-1
        @inbounds result += ackley_term(x[i], x[i+1])
    end
    return result
end

#############################
# Benchmark utility methods #
#############################
# Usage:
#
# # The values of the Dict are Arrays of time values where indices correspond to length(x)
# julia> t = bench_fad(ackley_sum, 10:10:100, 4) # benchmark ackley_sum where length(x) = 10:10:100, taking the minimum of 4 trials
# Dict{Symbol,Array{Float64,1}} with 3 entries: 
#   :gtimes => [6.184e-6,2.8753e-5,5.4635e-5,6.8362e-5,9.353e-5,0.000115482,0.00014288,0.00017434,0.000209135,0.0002428…
#   :htimes => [6.0639e-5,0.000191262,0.000569371,0.001274732,0.00284165,0.00518009,0.010288519,0.016470037,0.021582144…
#   :ftimes => [1.227e-6,1.888e-6,2.586e-6,3.264e-6,4.062e-6,4.943e-6,5.547e-6,6.325e-6,6.808e-6,7.728e-6]

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

