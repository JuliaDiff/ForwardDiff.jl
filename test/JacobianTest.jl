module JacobianTest

import Calculus

using Base.Test
using ForwardDiff
using ForwardDiff: default_value, KWARG_DEFAULTS

########################
# @jacobian/@jacobian! #
########################

const ALL = :(Val{$(default_value(KWARG_DEFAULTS, :allresults))}())
const CHUNK = :(Val{$(default_value(KWARG_DEFAULTS, :chunk))}())
const LEN = :(Val{$(default_value(KWARG_DEFAULTS, :len))}())
const MULTITHREAD = :(Val{$(default_value(KWARG_DEFAULTS, :multithread))}())

@test macroexpand(:(ForwardDiff.@jacobian(f, x))) == :(ForwardDiff.jacobian_entry_point($CHUNK, $LEN, $ALL, $MULTITHREAD, x, f))
@test macroexpand(:(ForwardDiff.@jacobian(f, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.jacobian_entry_point(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, f))
@test macroexpand(:(ForwardDiff.@jacobian(f, x; chunk=1, multithread=2))) == :(ForwardDiff.jacobian_entry_point(Val{1}(), $LEN, $ALL, Val{2}(), x, f))

@test macroexpand(:(ForwardDiff.@jacobian(f!, y, x))) == :(ForwardDiff.jacobian_entry_point($CHUNK, $LEN, $ALL, $MULTITHREAD, x, f!, y))
@test macroexpand(:(ForwardDiff.@jacobian(f!, y, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.jacobian_entry_point(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, f!, y))
@test macroexpand(:(ForwardDiff.@jacobian(f!, y, x; chunk=1, multithread=2))) == :(ForwardDiff.jacobian_entry_point(Val{1}(), $LEN, $ALL, Val{2}(), x, f!, y))

@test macroexpand(:(ForwardDiff.@jacobian!(out, f, x))) == :(ForwardDiff.jacobian_entry_point!($CHUNK, $LEN, $ALL, $MULTITHREAD, x, out, f))
@test macroexpand(:(ForwardDiff.@jacobian!(out, f, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.jacobian_entry_point!(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, out, f))
@test macroexpand(:(ForwardDiff.@jacobian!(out, f, x; chunk=1, multithread=2))) == :(ForwardDiff.jacobian_entry_point!(Val{1}(), $LEN, $ALL, Val{2}(), x, out, f))

@test macroexpand(:(ForwardDiff.@jacobian!(out, f!, y, x))) == :(ForwardDiff.jacobian_entry_point!($CHUNK, $LEN, $ALL, $MULTITHREAD, x, out, f!, y))
@test macroexpand(:(ForwardDiff.@jacobian!(out, f!, y, x, len=1, allresults=2, multithread=3, chunk=4))) == :(ForwardDiff.jacobian_entry_point!(Val{4}(), Val{1}(), Val{2}(), Val{3}(), x, out, f!, y))
@test macroexpand(:(ForwardDiff.@jacobian!(out, f!, y, x; chunk=1, multithread=2))) == :(ForwardDiff.jacobian_entry_point!(Val{1}(), $LEN, $ALL, Val{2}(), x, out, f!, y))

##################
# Test Functions #
##################

include(joinpath(dirname(@__FILE__), "utils.jl"))

function test_approx_eps(a::ForwardDiff.JacobianResult, b::ForwardDiff.JacobianResult)
    ForwardDiff.value(a) == ForwardDiff.value(b)
    test_approx_eps(ForwardDiff.jacobian(a), ForwardDiff.jacobian(b))
end

for f in VECTOR_TO_VECTOR_FUNCS
    valresult = f(X)
    jacresult = Calculus.jacobian(f, X, :forward)
    fullresult = ForwardDiff.JacobianChunkResult(valresult, jacresult)
    for c in CHUNK_SIZES
        ###################
        # single-threaded #
        ###################
        # @jacobian(f, x)
        g1 = x -> ForwardDiff.@jacobian(f, x; chunk = c, len = XLEN)
        g2 = x -> ForwardDiff.@jacobian(f, x; chunk = c, allresults = true)
        test_approx_eps(jacresult, g1(X))
        test_approx_eps(fullresult, g2(X))
        test_approx_eps(jacresult, ForwardDiff.@jacobian(f, X; chunk = c, len = XLEN))
        test_approx_eps(fullresult, ForwardDiff.@jacobian(f, X; chunk = c, allresults = true))
        # @jacobian!(out, f, x)
        g1! = (out, x) -> ForwardDiff.@jacobian!(out, f, x; chunk = c)
        g2! = (out, x) -> ForwardDiff.@jacobian!(out, f, x; chunk = c, len = XLEN, allresults = true)
        out1 = similar(X, XLEN, YLEN)
        out2 = similar(X, XLEN, YLEN)
        out3 = similar(X, XLEN, YLEN)
        out4 = similar(X, XLEN, YLEN)
        test_approx_eps(jacresult, g1!(out1, X))
        test_approx_eps(jacresult, out1)
        test_approx_eps(fullresult, g2!(out2, X))
        test_approx_eps(jacresult, out2)
        test_approx_eps(jacresult, ForwardDiff.@jacobian!(out3, f, X; chunk = c, len = XLEN))
        test_approx_eps(jacresult, out3)
        test_approx_eps(fullresult, ForwardDiff.@jacobian!(out4, f, X; chunk = c, allresults = true))
        test_approx_eps(jacresult, out4)
        if ForwardDiff.IS_MULTITHREADED_JULIA
            #################
            # multithreaded #
            #################
            # @jacobian(f, x)
            g1  = x -> ForwardDiff.@jacobian(f, x; multithread = true, chunk = c, len = XLEN)
            g2  = x -> ForwardDiff.@jacobian(f, x; multithread = true, chunk = c, allresults = true)
            test_approx_eps(jacresult, g1(X))
            test_approx_eps(fullresult, g2(X))
            test_approx_eps(jacresult, ForwardDiff.@jacobian(f, X; multithread = true, chunk = c, len = XLEN))
            test_approx_eps(fullresult, ForwardDiff.@jacobian(f, X; multithread = true, chunk = c, allresults = true))
            # @jacobian!(out, f, x)
            g1! = (out, x) -> ForwardDiff.@jacobian!(out, f, x; multithread = true, chunk = c)
            g2! = (out, x) -> ForwardDiff.@jacobian!(out, f, x; multithread = true, chunk = c, len = XLEN, allresults = true)
            out1 = similar(X, XLEN, YLEN)
            out2 = similar(X, XLEN, YLEN)
            out3 = similar(X, XLEN, YLEN)
            out4 = similar(X, XLEN, YLEN)
            test_approx_eps(jacresult, g1!(out1, X))
            test_approx_eps(jacresult, out1)
            test_approx_eps(fullresult, g2!(out2, X))
            test_approx_eps(jacresult, out2)
            test_approx_eps(jacresult, ForwardDiff.@jacobian!(out3, f, X; multithread = true, chunk = c, len = XLEN))
            test_approx_eps(jacresult, out3)
            test_approx_eps(fullresult, ForwardDiff.@jacobian!(out4, f, X; multithread = true, chunk = c, allresults = true))
            test_approx_eps(jacresult, out4)
        end
    end
end

for f! in VECTOR_TO_VECTOR_INPLACE_FUNCS
    # TODO
end

end # module
