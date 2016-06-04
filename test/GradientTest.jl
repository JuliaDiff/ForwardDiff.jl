module GradientTest

include(joinpath(dirname(@__FILE__), "TestFuncs.jl"))

import Calculus

using Base.Test
using ForwardDiff
using ForwardDiff: default_value, KWARG_DEFAULTS

########################
# @gradient/@gradient! #
########################

const ALL = :(Val{$(default_value(KWARG_DEFAULTS, :allresults))}())
const CHUNK = :(Val{$(default_value(KWARG_DEFAULTS, :chunk))}())
const LEN = :(Val{$(default_value(KWARG_DEFAULTS, :len))}())
const MULTITHREAD = :(Val{$(default_value(KWARG_DEFAULTS, :multithread))}())

@test macroexpand(:(ForwardDiff.@gradient(sin, x))) == :(ForwardDiff.gradient_entry_point($CHUNK, $LEN, $ALL, $MULTITHREAD, x, sin))
@test macroexpand(:(ForwardDiff.@gradient(sin, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.gradient_entry_point(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, sin))
@test macroexpand(:(ForwardDiff.@gradient(sin, x; chunk=1, multithread=2))) == :(ForwardDiff.gradient_entry_point(Val{1}(), $LEN, $ALL, Val{2}(), x, sin))

@test macroexpand(:(ForwardDiff.@gradient!(output, sin, x))) == :(ForwardDiff.gradient_entry_point!($CHUNK, $LEN, $ALL, $MULTITHREAD, x, output, sin))
@test macroexpand(:(ForwardDiff.@gradient!(output, sin, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.gradient_entry_point!(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, output, sin))
@test macroexpand(:(ForwardDiff.@gradient!(output, sin, x; chunk=1, multithread=2))) == :(ForwardDiff.gradient_entry_point!(Val{1}(), $LEN, $ALL, Val{2}(), x, output, sin))

##################
# Test Functions #
##################

const XLEN = 10
const X = rand(XLEN)
const GRADEPS = 1e-6

# There's going to be some approximation error, since we're testing
# against a result calculated via finite difference.
test_approx_grad(a::Array, b::Array) = @test_approx_eq_eps a b GRADEPS
test_approx_grad(a::Number, b::Number) = @test_approx_eq_eps a b GRADEPS

function test_approx_grad(a::ForwardDiff.GradientResult, b::ForwardDiff.GradientResult)
    test_approx_grad(ForwardDiff.value(a), ForwardDiff.value(b))
    test_approx_grad(ForwardDiff.gradient(a), ForwardDiff.gradient(b))
end

output() = similar(X)

for f in TestFuncs.VECTOR_TO_NUMBER_FUNCS
    valresult = f(X)
    gradresult = Calculus.gradient(f, X)
    fullresult = ForwardDiff.GradientChunkResult(valresult, gradresult)
    for c in (default_value(KWARG_DEFAULTS, :chunk), div(XLEN, 2), div(XLEN, 2) + 1, XLEN)
        ###################
        # single-threaded #
        ###################
        # @gradient(f)
        g1  = x -> ForwardDiff.@gradient(f, x; chunk = c, len = XLEN)
        g1! = (y, x) -> ForwardDiff.@gradient!(y, f, x; chunk = c)
        g2  = x -> ForwardDiff.@gradient(f, x; chunk = c, allresults = true)
        g2! = (y, x) -> ForwardDiff.@gradient!(y, f, x; chunk = c, len = XLEN, allresults = true)
        out1 = output()
        out2 = output()
        test_approx_grad(gradresult, g1(X))
        test_approx_grad(gradresult, g1!(out1, X))
        test_approx_grad(gradresult, out1)
        test_approx_grad(fullresult, g2(X))
        test_approx_grad(fullresult, g2!(out2, X))
        test_approx_grad(gradresult, out2)
        # @gradient(f, x)
        test_approx_grad(gradresult, ForwardDiff.@gradient(f, X; chunk = c, len = XLEN))
        test_approx_grad(fullresult, ForwardDiff.@gradient(f, X; chunk = c, allresults = true))
        # @gradient!(f, out, x)
        out3 = output()
        out4 = output()
        test_approx_grad(gradresult, ForwardDiff.@gradient!(out3, f, X; chunk = c, len = XLEN))
        test_approx_grad(gradresult, out3)
        test_approx_grad(fullresult, ForwardDiff.@gradient!(out4, f, X; chunk = c, allresults = true))
        test_approx_grad(gradresult, out4)
        if ForwardDiff.IS_MULTITHREADED_JULIA
            #################
            # multithreaded #
            #################
            # @gradient(f)
            g1  = x -> ForwardDiff.@gradient(f, x; multithread = true, chunk = c, len = XLEN)
            g1! = (y, x) -> ForwardDiff.@gradient!(y, f, x; multithread = true, chunk = c)
            g2  = x -> ForwardDiff.@gradient(f, x; multithread = true, chunk = c, allresults = true)
            g2! = (y, x) -> ForwardDiff.@gradient!(y, f, x; multithread = true, chunk = c, len = XLEN, allresults = true)
            out1 = output()
            out2 = output()
            test_approx_grad(gradresult, g1(X))
            test_approx_grad(gradresult, g1!(out1, X))
            test_approx_grad(gradresult, out1)
            test_approx_grad(fullresult, g2(X))
            test_approx_grad(fullresult, g2!(out2, X))
            test_approx_grad(gradresult, out2)
            # @gradient(f, x)
            test_approx_grad(gradresult, ForwardDiff.@gradient(f, X; multithread = true, chunk = c, len = XLEN))
            test_approx_grad(fullresult, ForwardDiff.@gradient(f, X; multithread = true, chunk = c, allresults = true))
            # @gradient!(f, out, x)
            out3 = output()
            out4 = output()
            test_approx_grad(gradresult, ForwardDiff.@gradient!(out3, f, X; multithread = true, chunk = c, len = XLEN))
            test_approx_grad(gradresult, out3)
            test_approx_grad(fullresult, ForwardDiff.@gradient!(out4, f, X; multithread = true, chunk = c, allresults = true))
            test_approx_grad(gradresult, out4)
        end
    end
end

end # module
