module GradientTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

#############################
# rosenbrock hardcoded test #
#############################

x = [0.1, 0.2, 0.3]
v = rosenbrock(x)
g = [-9.4, 15.6, 52.0]

for c in (Chunk{1}(), Chunk{2}(), Chunk{3}())

    # single-threaded #
    #-----------------#
    @test_approx_eq g ForwardDiff.gradient(rosenbrock, x, c)

    out = similar(x)
    ForwardDiff.gradient!(out, rosenbrock, x, c)
    @test_approx_eq out g

    out = GradientResult(x)
    ForwardDiff.gradient!(out, rosenbrock, x, c)
    @test_approx_eq ForwardDiff.value(out) v
    @test_approx_eq ForwardDiff.gradient(out) g

    # multithreaded #
    #---------------#
    if ForwardDiff.IS_MULTITHREADED_JULIA
        @test_approx_eq g ForwardDiff.gradient(rosenbrock, x, c; multithread = true)

        out = similar(x)
        ForwardDiff.gradient!(out, rosenbrock, x, c; multithread = true)
        @test_approx_eq out g

        out = GradientResult(x)
        ForwardDiff.gradient!(out, rosenbrock, x, c; multithread = true)
        @test_approx_eq ForwardDiff.value(out) v
        @test_approx_eq ForwardDiff.gradient(out) g
    end
end

########################
# test vs. Calculus.jl #
########################

for f in VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    @test_approx_eq_eps g Calculus.gradient(f, X) FINITEDIFF_ERROR
    for c in CHUNK_SIZES
        for usecache in (true, false)
            println("  ...testing $f with (chunk size = $c) and (usecache = $usecache)")
            chunk = Chunk{c}()

            # single-threaded #
            #-----------------#
            out = ForwardDiff.gradient(f, X, chunk; usecache = usecache)
            @test_approx_eq out g

            out = similar(X)
            ForwardDiff.gradient!(out, f, X, chunk; usecache = usecache)
            @test_approx_eq out g

            out = GradientResult(X)
            ForwardDiff.gradient!(out, f, X, chunk; usecache = usecache)
            @test_approx_eq ForwardDiff.value(out) v
            @test_approx_eq ForwardDiff.gradient(out) g

            # multithreaded #
            #---------------#
            if ForwardDiff.IS_MULTITHREADED_JULIA
                out = ForwardDiff.gradient(f, X, chunk; multithread = true, usecache = usecache)
                @test_approx_eq out g

                out = similar(X)
                ForwardDiff.gradient!(out, f, X, chunk; multithread = true, usecache = usecache)
                @test_approx_eq out g

                out = GradientResult(X)
                ForwardDiff.gradient!(out, f, X, chunk; multithread = true, usecache = usecache)
                @test_approx_eq ForwardDiff.value(out) v
                @test_approx_eq ForwardDiff.gradient(out) g
            end
        end
    end
end

end # module
