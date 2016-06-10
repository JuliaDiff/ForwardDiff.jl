module GradientTest

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

@test macroexpand(:(ForwardDiff.@gradient(f, x))) == :(ForwardDiff.gradient_entry_point($CHUNK, $LEN, $ALL, $MULTITHREAD, x, f))
@test macroexpand(:(ForwardDiff.@gradient(f, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.gradient_entry_point(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, f))
@test macroexpand(:(ForwardDiff.@gradient(f, x; chunk=1, multithread=2))) == :(ForwardDiff.gradient_entry_point(Val{1}(), $LEN, $ALL, Val{2}(), x, f))

@test macroexpand(:(ForwardDiff.@gradient!(out, f, x))) == :(ForwardDiff.gradient_entry_point!($CHUNK, $LEN, $ALL, $MULTITHREAD, x, out, f))
@test macroexpand(:(ForwardDiff.@gradient!(out, f, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.gradient_entry_point!(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, out, f))
@test macroexpand(:(ForwardDiff.@gradient!(out, f, x; chunk=1, multithread=2))) == :(ForwardDiff.gradient_entry_point!(Val{1}(), $LEN, $ALL, Val{2}(), x, out, f))

##################
# Test Functions #
##################

include(joinpath(dirname(@__FILE__), "utils.jl"))

function test_approx_eps(a::ForwardDiff.GradientResult, b::ForwardDiff.GradientResult)
    ForwardDiff.value(a) == ForwardDiff.value(b)
    test_approx_eps(ForwardDiff.gradient(a), ForwardDiff.gradient(b))
end

for f in VECTOR_TO_NUMBER_FUNCS
    valresult = f(X)
    gradresult = Calculus.gradient(f, X)
    fullresult = ForwardDiff.GradientChunkResult(valresult, gradresult)
    for c in CHUNK_SIZES
        ###################
        # single-threaded #
        ###################
        # @gradient(f, x)
        g1 = x -> ForwardDiff.@gradient(f, x; chunk = c, len = XLEN)
        g2 = x -> ForwardDiff.@gradient(f, x; chunk = c, allresults = true)
        test_approx_eps(gradresult, g1(X))
        test_approx_eps(fullresult, g2(X))
        test_approx_eps(gradresult, ForwardDiff.@gradient(f, X; chunk = c, len = XLEN))
        test_approx_eps(fullresult, ForwardDiff.@gradient(f, X; chunk = c, allresults = true))
        # @gradient!(out, f, x)
        g1! = (out, x) -> ForwardDiff.@gradient!(out, f, x; chunk = c)
        g2! = (out, x) -> ForwardDiff.@gradient!(out, f, x; chunk = c, len = XLEN, allresults = true)
        out1 = similar(X)
        out2 = similar(X)
        out3 = similar(X)
        out4 = similar(X)
        test_approx_eps(gradresult, g1!(out1, X))
        test_approx_eps(gradresult, out1)
        test_approx_eps(fullresult, g2!(out2, X))
        test_approx_eps(gradresult, out2)
        test_approx_eps(gradresult, ForwardDiff.@gradient!(out3, f, X; chunk = c, len = XLEN))
        test_approx_eps(gradresult, out3)
        test_approx_eps(fullresult, ForwardDiff.@gradient!(out4, f, X; chunk = c, allresults = true))
        test_approx_eps(gradresult, out4)
        if ForwardDiff.IS_MULTITHREADED_JULIA
            #################
            # multithreaded #
            #################
            # @gradient(f, x)
            g1  = x -> ForwardDiff.@gradient(f, x; multithread = true, chunk = c, len = XLEN)
            g2  = x -> ForwardDiff.@gradient(f, x; multithread = true, chunk = c, allresults = true)
            test_approx_eps(gradresult, g1(X))
            test_approx_eps(fullresult, g2(X))
            test_approx_eps(gradresult, ForwardDiff.@gradient(f, X; multithread = true, chunk = c, len = XLEN))
            test_approx_eps(fullresult, ForwardDiff.@gradient(f, X; multithread = true, chunk = c, allresults = true))
            # @gradient!(out, f, x)
            g1! = (out, x) -> ForwardDiff.@gradient!(out, f, x; multithread = true, chunk = c)
            g2! = (out, x) -> ForwardDiff.@gradient!(out, f, x; multithread = true, chunk = c, len = XLEN, allresults = true)
            out1 = similar(X)
            out2 = similar(X)
            out3 = similar(X)
            out4 = similar(X)
            test_approx_eps(gradresult, g1!(out1, X))
            test_approx_eps(gradresult, out1)
            test_approx_eps(fullresult, g2!(out2, X))
            test_approx_eps(gradresult, out2)
            test_approx_eps(gradresult, ForwardDiff.@gradient!(out3, f, X; multithread = true, chunk = c, len = XLEN))
            test_approx_eps(gradresult, out3)
            test_approx_eps(fullresult, ForwardDiff.@gradient!(out4, f, X; multithread = true, chunk = c, allresults = true))
            test_approx_eps(gradresult, out4)
        end
    end
end

end # module
