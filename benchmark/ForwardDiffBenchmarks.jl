module ForwardDiffBenchmarks

using ForwardDiff
using BenchmarkTools

include(joinpath(Pkg.dir("ForwardDiff"), "test", "TestFuncs.jl"))

name(f) = last(split(string(f), '.'))

const GROUPS = BenchmarkTools.GroupCollection()

xs = map(n -> rand(MersenneTwister(1), n), (12, 120, 1200))
chunks = 1:10

val = addgroup!(GROUPS, "value")
grad = addgroup!(GROUPS, "gradient")

for f in TestFuncs.VECTOR_TO_NUMBER_FUNCS
    for x in xs
        n = length(x)
        val[name(f), n] = @benchmarkable $(f)($x)
        for c in chunks
            grad[name(f), n, c] = @benchmarkable ForwardDiff.@gradient($f, $x, chunk = $c, len = $n)
        end
    end
end

end # module
