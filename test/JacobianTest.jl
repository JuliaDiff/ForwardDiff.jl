module JacobianTest

import Calculus

using Base.Test
using ForwardDiff

##################
# Test Functions #
##################

include(joinpath(dirname(@__FILE__), "utils.jl"))

function test_approx_eps(a::ForwardDiff.JacobianResult, b::ForwardDiff.JacobianResult)
    ForwardDiff.value(a) == ForwardDiff.value(b)
    test_approx_eps(ForwardDiff.jacobian(a), ForwardDiff.jacobian(b))
end

for f in VECTOR_TO_VECTOR_FUNCS
    jacresult = Calculus.jacobian(f, X, :forward)
    fullresult = ForwardDiff.JacobianResult(f(X), jacresult)
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size $c")
        chunk = Chunk{c}()
        ###################
        # single-threaded #
        ###################
        out = ForwardDiff.jacobian(f, X, chunk)
        test_approx_eps(jacresult, out)

        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f, X, chunk)
        test_approx_eps(jacresult, out)

        out = JacobianResult(similar(Y), similar(Y, length(Y), length(X)))
        ForwardDiff.jacobian!(out, f, X, chunk)
        test_approx_eps(fullresult, out)

        #################
        # multithreaded #
        #################
        if ForwardDiff.IS_MULTITHREADED_JULIA
            out = ForwardDiff.jacobian(f, X, chunk; multithread = true)
            test_approx_eps(jacresult, out)

            out = similar(Y)
            ForwardDiff.jacobian!(out, f, X, chunk; multithread = true)
            test_approx_eps(jacresult, out)

            out = JacobianResult(similar(Y), similar(Y, length(Y), length(X)))
            ForwardDiff.jacobian!(out, f, X, chunk; multithread = true)
            test_approx_eps(fullresult, out)
        end
    end
end

for f! in VECTOR_TO_VECTOR_INPLACE_FUNCS
    # TODO
end

end # module
