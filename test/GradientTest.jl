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
    gradresult = Calculus.gradient(f, X)
    fullresult = GradientResult(f(X), gradresult)
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size $c")
        chunk = Chunk{c}()
        ###################
        # single-threaded #
        ###################
        out = ForwardDiff.gradient(f, X, chunk)
        test_approx_eps(gradresult, out)

        out = similar(X)
        ForwardDiff.gradient!(out, f, X, chunk)
        test_approx_eps(gradresult, out)

        out = GradientResult(first(X), similar(X))
        ForwardDiff.gradient!(out, f, X, chunk)
        test_approx_eps(fullresult, out)

        #################
        # multithreaded #
        #################
        if ForwardDiff.IS_MULTITHREADED_JULIA
            out = ForwardDiff.gradient(f, X, chunk; multithread = true)
            test_approx_eps(gradresult, out)

            out = similar(X)
            ForwardDiff.gradient!(out, f, X, chunk; multithread = true)
            test_approx_eps(gradresult, out)

            out = GradientResult(first(X), similar(X))
            ForwardDiff.gradient!(out, f, X, chunk; multithread = true)
            test_approx_eps(fullresult, out)
        end
    end
end

end # module
