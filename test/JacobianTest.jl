module JacobianTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

##################
# hardcoded test #
##################

f! = (y, x) -> begin
    y[1] = x[1] * x[2]
    y[1] *= sin(x[3]^2)
    y[2] = y[1] + x[3]
    y[3] = y[1] / y[2]
    y[4] = x[3]
    return nothing
end
f = x -> (y = zeros(promote_type(eltype(x), Float64), 4); f!(y, x); return y)
y = zeros(4)
x = [1, 2, 3]
v = f(x)
j = [0.8242369704835132  0.4121184852417566  -10.933563142616123
     0.8242369704835132  0.4121184852417566  -9.933563142616123
     0.169076696546684   0.084538348273342   -2.299173530851733
     0.0                 0.0                 1.0]

for c in (Chunk{1}(), Chunk{2}(), Chunk{3}())
    # testing f(x)
    @test_approx_eq j ForwardDiff.jacobian(f, x, c)

    out = zeros(4, 3)
    ForwardDiff.jacobian!(out, f, x, c)
    @test_approx_eq out j

    # testing f!(y, x)
    @test_approx_eq j ForwardDiff.jacobian(f!, y, x, c)
    @test_approx_eq v y

    fill!(y, 0.0)
    out = zeros(4, 3)
    ForwardDiff.jacobian!(out, f!, y, x, c)
    @test_approx_eq out j
    @test_approx_eq y v

    out = JacobianResult(zeros(4), zeros(4, 3))
    ForwardDiff.jacobian!(out, f!, x, c)
    @test_approx_eq ForwardDiff.value(out) v
    @test_approx_eq ForwardDiff.jacobian(out) j
end

########################
# test vs. Calculus.jl #
########################

for (f!, f) in VECTOR_TO_VECTOR_FUNCS
    v = f(X)
    j = ForwardDiff.jacobian(f, X)
    @test_approx_eq_eps j Calculus.jacobian(f, X, :forward) FINITEDIFF_ERROR
    for c in CHUNK_SIZES
        chunk = Chunk{c}()

        # testing f(x)
        println("  ...testing $f with chunk size $c")
        out = ForwardDiff.jacobian(f, X, chunk)
        @test_approx_eq out j

        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f, X, chunk)
        @test_approx_eq out j

        # testing f!(y, x)
        println("  ...testing $(f!) with chunk size $c")
        y = zeros(Y)
        out = ForwardDiff.jacobian(f!, y, X, chunk)
        @test_approx_eq y v
        @test_approx_eq out j

        y = zeros(Y)
        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f!, y, X, chunk)
        @test_approx_eq y v
        @test_approx_eq out j

        out = JacobianResult(zeros(Y), similar(Y, length(Y), length(X)))
        ForwardDiff.jacobian!(out, f!, X, chunk)
        @test_approx_eq ForwardDiff.value(out) v
        @test_approx_eq ForwardDiff.jacobian(out) j
    end
end

end # module
