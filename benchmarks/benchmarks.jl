using ForwardDiff, DiffTests, DiffResults
using BenchmarkTools

include(joinpath(dirname(dirname(@__FILE__)), "test", "utils.jl"))

name(f) = last(split(string(f), '.'))

const SUITE = BenchmarkGroup()

const vecs = map(n -> rand(MersenneTwister(1), n), (10, 100, 1000))
const mats = map(n -> rand(MersenneTwister(1), n, n), (5, 16, 32))

const value_group = addgroup!(SUITE, "value")
const derivative_group = addgroup!(SUITE, "derivative")
const gradient_group = addgroup!(SUITE, "gradient")
const jacobian_group = addgroup!(SUITE, "jacobian")
const hessian_group = addgroup!(SUITE, "hessian")

for f in (DiffTests.NUMBER_TO_NUMBER_FUNCS..., DiffTests.NUMBER_TO_ARRAY_FUNCS...)
    x = 1.0
    y = f(x)

    value_group[name(f)] = @benchmarkable $(f)($x)

    out = isa(y, Number) ? DiffResults.DiffResult(y, y) : DiffResults.DiffResult(similar(y), similar(y))
    derivative_group[name(f)] = @benchmarkable ForwardDiff.derivative!($out, $f, $x)
end

for f in (DiffTests.VECTOR_TO_NUMBER_FUNCS..., DiffTests.MATRIX_TO_NUMBER_FUNCS...)
    fval = addgroup!(value_group, name(f))
    fgrad = addgroup!(gradient_group, name(f))
    fhess = addgroup!(hessian_group, name(f))
    arrs = in(f, DiffTests.VECTOR_TO_NUMBER_FUNCS) ? vecs : mats
    for x in arrs
        y = f(x)

        fval[length(x)] = @benchmarkable $(f)($x)

        gout = DiffResults.DiffResult(y, similar(x, typeof(y)))
        gcfg = ForwardDiff.GradientConfig(nothing, x)
        fgrad[length(x)] = @benchmarkable ForwardDiff.gradient!($gout, $f, $x, $gcfg)

        hout = DiffResults.DiffResult(y, similar(x, typeof(y)), similar(x, typeof(y), length(x), length(x)))
        hcfg = ForwardDiff.HessianConfig(nothing, hout, x)
        fhess[length(x)] = @benchmarkable ForwardDiff.hessian!($hout, $f, $x, $hcfg)
    end
end

for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
    fval = addgroup!(value_group, name(f))
    fjac = addgroup!(jacobian_group, name(f))
    for x in mats
        y = f(x)
        fval[length(x)] = @benchmarkable $(f)($x)

        out = DiffResults.JacobianResult(y, x)
        cfg = ForwardDiff.JacobianConfig(nothing, y, x)
        fjac[length(x)] = @benchmarkable ForwardDiff.jacobian!($out, $f, $y, $x, $cfg)
    end
end
