module ForwardDiffBenchmarks

using ForwardDiff
using BenchmarkTools

include(joinpath(dirname(dirname(@__FILE__)), "test", "utils.jl"))

name(f) = last(split(string(f), '.'))

const SUITE = BenchmarkGroup()

xs = map(n -> rand(MersenneTwister(1), n), (10, 100, 1000, 10000))
chunk_sizes = 1:10

val = addgroup!(SUITE, "value")
deriv = addgroup!(SUITE, "derivative")
grad = addgroup!(SUITE, "gradient")
jac = addgroup!(SUITE, "jacobian")
hess = addgroup!(SUITE, "hessian")

for f in (NUMBER_TO_NUMBER_FUNCS..., NUMBER_TO_ARRAY_FUNCS...)
    x = 1
    val[name(f)] = @benchmarkable $(f)($x)
    deriv[name(f)] = @benchmarkable ForwardDiff.derivative($f, $x)
end

for f in VECTOR_TO_NUMBER_FUNCS
    fval = addgroup!(val, name(f))
    fgrad = addgroup!(grad, name(f))
    fhess = addgroup!(hess, name(f))
    for x in xs
        xlen = length(x)
        fval[xlen] = @benchmarkable $(f)($x)
        for c in chunk_sizes
            fgrad[xlen, c] = @benchmarkable ForwardDiff.gradient($f, $x, $(Chunk{c}()))
            fhess[xlen, c] = @benchmarkable ForwardDiff.hessian($f, $x, $(Chunk{c}()))
        end
    end
end

for (f!, f) in VECTOR_TO_VECTOR_FUNCS
    fval = addgroup!(val, name(f))
    fjac = addgroup!(jac, name(f))
    for x in xs
        xlen = length(x)
        fval[xlen] = @benchmarkable $(f)($x)
        for c in chunk_sizes
            fjac[xlen, c] = @benchmarkable ForwardDiff.jacobian($f, $x, $(Chunk{c}()))
        end
    end
end

function runafew(seconds = 1)
    result = BenchmarkGroup()

    deriv = addgroup!(result, "derivative")
    tune!(SUITE["derivative"], seconds = seconds) # fast enough to require tuning
    for (f, b) in SUITE["derivative"]
        deriv[f] = run(b; seconds = seconds)
    end

    grad = addgroup!(result, "gradient")
    for (f, group) in SUITE["gradient"]
        grad[f] = run(group[1000, 10]; seconds = seconds)
    end

    hess = addgroup!(result, "hessian")
    for (f, group) in SUITE["hessian"]
        hess[f] = run(group[100, 10]; seconds = seconds)
    end

    jac = addgroup!(result, "jacobian")
    for (f, group) in SUITE["jacobian"]
        jac[f] = run(group[1000, 10]; seconds = seconds)
    end

    return result
end

end # module
