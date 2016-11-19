module GradientTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

##################
# hardcoded test #
##################

f = DiffBase.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]

for c in (1, 2, 3)
    println("  ...running hardcoded test with chunk size = $c")
    cfg = ForwardDiff.GradientConfig{c}(x)

    # single-threaded #
    #-----------------#
    @test_approx_eq g ForwardDiff.gradient(f, x, cfg)
    @test_approx_eq g ForwardDiff.gradient(f, x)

    out = similar(x)
    ForwardDiff.gradient!(out, f, x, cfg)
    @test_approx_eq out g

    out = similar(x)
    ForwardDiff.gradient!(out, f, x)
    @test_approx_eq out g

    out = DiffBase.GradientResult(x)
    ForwardDiff.gradient!(out, f, x, cfg)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.gradient(out) g

    out = DiffBase.GradientResult(x)
    ForwardDiff.gradient!(out, f, x)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.gradient(out) g

    # multithreaded #
    #---------------#
    if ForwardDiff.IS_MULTITHREADED_JULIA
        multi_cfg = ForwardDiff.MultithreadConfig(cfg)

        @test_approx_eq g ForwardDiff.gradient(f, x, multi_cfg)

        out = similar(x)
        ForwardDiff.gradient!(out, f, x, multi_cfg)
        @test_approx_eq out g

        out = DiffBase.GradientResult(x)
        ForwardDiff.gradient!(out, f, x, multi_cfg)
        @test_approx_eq DiffBase.value(out) v
        @test_approx_eq DiffBase.gradient(out) g
    end
end

########################
# test vs. Calculus.jl #
########################

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    @test_approx_eq_eps g Calculus.gradient(f, X) FINITEDIFF_ERROR
    for c in CHUNK_SIZES
        println("  ...testing $f with chunk size = $c")
        cfg = ForwardDiff.GradientConfig{c}(X)

        # single-threaded #
        #-----------------#
        out = ForwardDiff.gradient(f, X, cfg)
        @test_approx_eq out g

        out = similar(X)
        ForwardDiff.gradient!(out, f, X, cfg)
        @test_approx_eq out g

        out = DiffBase.GradientResult(X)
        ForwardDiff.gradient!(out, f, X, cfg)
        @test_approx_eq DiffBase.value(out) v
        @test_approx_eq DiffBase.gradient(out) g

        # multithreaded #
        #---------------#
        if ForwardDiff.IS_MULTITHREADED_JULIA
            multi_cfg = ForwardDiff.MultithreadConfig(cfg)

            out = ForwardDiff.gradient(f, X, multi_cfg)
            @test_approx_eq out g

            out = similar(X)
            ForwardDiff.gradient!(out, f, X, multi_cfg)
            @test_approx_eq out g

            out = DiffBase.GradientResult(X)
            ForwardDiff.gradient!(out, f, X, multi_cfg)
            @test_approx_eq DiffBase.value(out) v
            @test_approx_eq DiffBase.gradient(out) g
        end
    end
end

end # module
