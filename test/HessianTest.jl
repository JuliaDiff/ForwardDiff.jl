module HessianTest

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
h = [-66.0  -40.0    0.0;
     -40.0  130.0  -80.0;
       0.0  -80.0  200.0]

for c in (Chunk{1}(), Chunk{2}(), Chunk{3}())

    # single-threaded #
    #-----------------#
    @test_approx_eq h ForwardDiff.hessian(rosenbrock, x, c)

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, rosenbrock, x, c)
    @test_approx_eq out h

    out = HessianResult(x)
    ForwardDiff.hessian!(out, rosenbrock, x, c)
    @test_approx_eq ForwardDiff.value(out) v
    @test_approx_eq ForwardDiff.gradient(out) g
    @test_approx_eq ForwardDiff.hessian(out) h

    # multithreaded #
    #---------------#
    if ForwardDiff.IS_MULTITHREADED_JULIA
        @test_approx_eq h ForwardDiff.hessian(rosenbrock, x, c; multithread = true)

        out = similar(x)
        ForwardDiff.hessian!(out, rosenbrock, x, c; multithread = true)
        @test_approx_eq out h

        out = HessianResult(x)
        ForwardDiff.hessian!(out, rosenbrock, x, c; multithread = true)
        @test_approx_eq ForwardDiff.value(out) v
        @test_approx_eq ForwardDiff.gradient(out) g
        @test_approx_eq ForwardDiff.hessian(out) h
    end
end

########################
# test vs. Calculus.jl #
########################

for f in VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    h = ForwardDiff.hessian(f, X)
    # finite difference approximation error is really bad for Hessians...
    @test_approx_eq_eps h Calculus.hessian(f, X) 0.01
    for c in CHUNK_SIZES
        for usecache in (true, false)
            println("  ...testing $f with (chunk size = $c) and (usecache = $usecache)")
            chunk = Chunk{c}()

            # single-threaded #
            #-----------------#
            out = ForwardDiff.hessian(f, X, chunk; usecache = usecache)
            @test_approx_eq out h

            out = similar(X, length(X), length(X))
            ForwardDiff.hessian!(out, f, X, chunk; usecache = usecache)
            @test_approx_eq out h

            out = HessianResult(X)
            ForwardDiff.hessian!(out, f, X, chunk; usecache = usecache)
            @test_approx_eq ForwardDiff.value(out) v
            @test_approx_eq ForwardDiff.gradient(out) g
            @test_approx_eq ForwardDiff.hessian(out) h

            # multithreaded #
            #---------------#
            if ForwardDiff.IS_MULTITHREADED_JULIA
                out = ForwardDiff.hessian(f, X, chunk; multithread = true, usecache = usecache)
                @test_approx_eq out h

                out = similar(X, length(X), length(X))
                ForwardDiff.hessian!(out, f, X, chunk; multithread = true, usecache = usecache)
                @test_approx_eq out h

                out = HessianResult(X)
                ForwardDiff.hessian!(out, f, X, chunk; multithread = true, usecache = usecache)
                @test_approx_eq ForwardDiff.value(out) v
                @test_approx_eq ForwardDiff.gradient(out) g
                @test_approx_eq ForwardDiff.hessian(out) h
            end
        end
    end
end

end # module
