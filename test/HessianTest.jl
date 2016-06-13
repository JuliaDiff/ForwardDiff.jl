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
    forwarddiff_hess = ForwardDiff.hessian(f, X)
    calculus_hess = Calculus.hessian(f, X)
    calculus_result = HessianResult(f(X), ForwardDiff.gradient(f, X), calculus_hess)
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size $c")
        chunk = Chunk{c}()

        out = ForwardDiff.hessian(f, X, chunk)
        test_approx_eps(calculus_hess, out)
        @test out == forwarddiff_hess

        out = similar(X, length(X), length(X))
        ForwardDiff.hessian!(out, f, X, chunk)
        test_approx_eps(calculus_hess, out)
        @test out == forwarddiff_hess

        out = HessianResult(X)
        ForwardDiff.hessian!(out, f, X, chunk)
        test_approx_eps(calculus_result, out)
        @test ForwardDiff.hessian(out) == forwarddiff_hess
    end
end

end # module
