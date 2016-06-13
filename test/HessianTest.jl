module HessianTest

import Calculus

using Base.Test
using ForwardDiff

##################
# Test Functions #
##################

include(joinpath(dirname(@__FILE__), "utils.jl"))

function test_approx_eps(a::ForwardDiff.HessianResult, b::ForwardDiff.HessianResult)
    ForwardDiff.value(a) == ForwardDiff.value(b)
    ForwardDiff.gradient(a) == ForwardDiff.gradient(b)
    test_approx_eps(ForwardDiff.hessian(a), ForwardDiff.hessian(b))
end

for f in VECTOR_TO_NUMBER_FUNCS
    hessresult = Calculus.hessian(f, X)
    fullresult = HessianResult(f(X), ForwardDiff.gradient(f, X), hessresult)
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size $c")
        chunk = Chunk{c}()

        out = ForwardDiff.hessian(f, X, chunk)
        test_approx_eps(hessresult, out)

        out = similar(X, length(x), length(x))
        ForwardDiff.hessian!(out, f, X, chunk)
        test_approx_eps(hessresult, out)

        out = HessianResult(X)
        ForwardDiff.hessian!(out, f, X, chunk)
        test_approx_eps(fullresult, out)
    end
end

end # module
