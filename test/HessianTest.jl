module HessianTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

#############################
# rosenbrock hardcoded test #
#############################

f = DiffBase.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]
h = [-66.0  -40.0    0.0;
     -40.0  130.0  -80.0;
       0.0  -80.0  200.0]

for c in (1, 2, 3)
    println("  ...running hardcoded test with chunk size = $c")
    outer_opts = ForwardDiff.Options{c}(x)
    inner_opts = ForwardDiff.Options(outer_opts)
    outer_opts2 = ForwardDiff.Options{c}(x, x)
    inner_opts2 = ForwardDiff.Options(outer_opts2)

    # single-threaded #
    #-----------------#
    @test_approx_eq h ForwardDiff.hessian(f, x)
    @test_approx_eq h ForwardDiff.hessian(f, x, outer_opts)
    @test_approx_eq h ForwardDiff.hessian(f, x, outer_opts, inner_opts)

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x)
    @test_approx_eq out h

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x, outer_opts)
    @test_approx_eq out h

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x, outer_opts, inner_opts)
    @test_approx_eq out h

    out = HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.gradient(out) g
    @test_approx_eq DiffBase.hessian(out) h

    out = HessianResult(x)
    ForwardDiff.hessian!(out, f, x, outer_opts2)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.gradient(out) g
    @test_approx_eq DiffBase.hessian(out) h

    out = HessianResult(x)
    ForwardDiff.hessian!(out, f, x, outer_opts2, inner_opts2)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.gradient(out) g
    @test_approx_eq DiffBase.hessian(out) h

    # multithreaded #
    #---------------#
    if ForwardDiff.IS_MULTITHREADED_JULIA
        multi_opts = ntuple(i -> copy(inner_opts), ForwardDiff.NTHREADS)
        multi_opts2 = ntuple(i -> copy(inner_opts2), ForwardDiff.NTHREADS)

        @test_approx_eq h ForwardDiff.hessian(f, x, outer_opts, multi_opts)

        out = similar(x, 3, 3)
        ForwardDiff.hessian!(out, f, x, outer_opts, multi_opts)
        @test_approx_eq out h

        out = HessianResult(x)
        ForwardDiff.hessian!(out, f, x, outer_opts2, multi_opts2)
        @test_approx_eq DiffBase.value(out) v
        @test_approx_eq DiffBase.gradient(out) g
        @test_approx_eq DiffBase.hessian(out) h
    end
end

########################
# test vs. Calculus.jl #
########################

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    h = ForwardDiff.hessian(f, X)
    # finite difference approximation error is really bad for Hessians...
    @test_approx_eq_eps h Calculus.hessian(f, X) 0.01
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size = $c")
        outer_opts = ForwardDiff.Options{c}(X)
        inner_opts = ForwardDiff.Options(outer_opts)
        outer_opts2 = ForwardDiff.Options{c}(X, X)
        inner_opts2 = ForwardDiff.Options(outer_opts2)

        # single-threaded #
        #-----------------#
        out = ForwardDiff.hessian(f, X, outer_opts, inner_opts)
        @test_approx_eq out h

        out = similar(X, length(X), length(X))
        ForwardDiff.hessian!(out, f, X, outer_opts, inner_opts)
        @test_approx_eq out h

        out = HessianResult(X)
        ForwardDiff.hessian!(out, f, X, outer_opts2, inner_opts2)
        @test_approx_eq DiffBase.value(out) v
        @test_approx_eq DiffBase.gradient(out) g
        @test_approx_eq DiffBase.hessian(out) h

        # multithreaded #
        #---------------#
        if ForwardDiff.IS_MULTITHREADED_JULIA
            multi_opts = ntuple(i -> copy(inner_opts), ForwardDiff.NTHREADS)
            multi_opts2 = ntuple(i -> copy(inner_opts2), ForwardDiff.NTHREADS)

            out = ForwardDiff.hessian(f, X, outer_opts, multi_opts)
            @test_approx_eq out h

            out = similar(X, length(X), length(X))
            ForwardDiff.hessian!(out, f, X, outer_opts, multi_opts)
            @test_approx_eq out h

            out = HessianResult(X)
            ForwardDiff.hessian!(out, f, X, outer_opts2, multi_opts2)
            @test_approx_eq DiffBase.value(out) v
            @test_approx_eq DiffBase.gradient(out) g
            @test_approx_eq DiffBase.hessian(out) h
        end
    end
end

end # module
