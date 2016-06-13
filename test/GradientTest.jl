module GradientTest

import Calculus

using Base.Test
using ForwardDiff

##################
# Test Functions #
##################

include(joinpath(dirname(@__FILE__), "utils.jl"))

function test_approx_eps(a::ForwardDiff.GradientResult, b::ForwardDiff.GradientResult)
    ForwardDiff.value(a) == ForwardDiff.value(b)
    test_approx_eps(ForwardDiff.gradient(a), ForwardDiff.gradient(b))
end

for f in VECTOR_TO_NUMBER_FUNCS
    forwarddiff_grad = ForwardDiff.gradient(f, X)
    calculus_grad = Calculus.gradient(f, X)
    calculus_result = GradientResult(f(X), calculus_grad)
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size $c")
        chunk = Chunk{c}()
        ###################
        # single-threaded #
        ###################
        out = ForwardDiff.gradient(f, X, chunk)
        test_approx_eps(calculus_grad, out)
        @test out == forwarddiff_grad

        out = similar(X)
        ForwardDiff.gradient!(out, f, X, chunk)
        test_approx_eps(calculus_grad, out)
        @test out == forwarddiff_grad

        out = GradientResult(X)
        ForwardDiff.gradient!(out, f, X, chunk)
        test_approx_eps(calculus_result, out)
        @test ForwardDiff.gradient(out) == forwarddiff_grad

        #################
        # multithreaded #
        #################
        if ForwardDiff.IS_MULTITHREADED_JULIA
            out = ForwardDiff.gradient(f, X, chunk; multithread = true)
            test_approx_eps(calculus_grad, out)
            @test out == forwarddiff_grad

            out = similar(X)
            ForwardDiff.gradient!(out, f, X, chunk; multithread = true)
            test_approx_eps(calculus_grad, out)
            @test out == forwarddiff_grad

            out = GradientResult(X)
            ForwardDiff.gradient!(out, f, X, chunk; multithread = true)
            test_approx_eps(calculus_result, out)
            @test ForwardDiff.gradient(out) == forwarddiff_grad
        end
    end
end

end # module
