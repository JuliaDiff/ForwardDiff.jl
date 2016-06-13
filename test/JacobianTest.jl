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

for (f!, f) in VECTOR_TO_VECTOR_FUNCS
    valresult = f(X)
    jacresult = Calculus.jacobian(f, X, :forward)
    fullresult = JacobianResult(valresult, jacresult)
    for c in CHUNK_SIZES
        chunk = Chunk{c}()

        # testing f(x)
        println("  ...testing $f with chunk size $c")
        out = ForwardDiff.jacobian(f, X, chunk)
        test_approx_eps(jacresult, out)

        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f, X, chunk)
        test_approx_eps(jacresult, out)

        # testing f!(y, x)
        println("  ...testing $(f!) with chunk size $c")
        y = zeros(Y)
        out = ForwardDiff.jacobian(f!, y, X, chunk)
        test_approx_eps(valresult, y)
        test_approx_eps(jacresult, out)

        y = zeros(Y)
        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f!, y, X, chunk)
        test_approx_eps(valresult, y)
        test_approx_eps(jacresult, out)

        out = JacobianResult(zeros(Y), similar(Y, length(Y), length(X)))
        ForwardDiff.jacobian!(out, f!, X, chunk)
        test_approx_eps(fullresult, out)
    end
end

end # module
