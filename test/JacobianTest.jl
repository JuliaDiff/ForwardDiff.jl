module JacobianTest

import Calculus

using Base.Test
using ForwardDiff
using ForwardDiff: Options

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
x = [1, 2, 3]
v = f(x)
j = [0.8242369704835132  0.4121184852417566  -10.933563142616123
     0.8242369704835132  0.4121184852417566  -9.933563142616123
     0.169076696546684   0.084538348273342   -2.299173530851733
     0.0                 0.0                 1.0]

for c in (1, 2, 3)
    println("  ...running hardcoded tests with chunk size $c")
    opts = Options{c}(x)
    yopts = Options{c}(zeros(4), x)

    # testing f(x)
    @test_approx_eq j ForwardDiff.jacobian(f, x, opts)
    @test_approx_eq j ForwardDiff.jacobian(f, x)

    out = zeros(4, 3)
    ForwardDiff.jacobian!(out, f, x, opts)
    @test_approx_eq out j

    out = zeros(4, 3)
    ForwardDiff.jacobian!(out, f, x)
    @test_approx_eq out j

    out = JacobianResult(zeros(4), zeros(3))
    ForwardDiff.jacobian!(out, f, x, Options(x))
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.jacobian(out) j

    # testing f!(y, x)
    y = zeros(4)
    @test_approx_eq j ForwardDiff.jacobian(f!, y, x, yopts)
    @test_approx_eq v y

    y = zeros(4)
    @test_approx_eq j ForwardDiff.jacobian(f!, y, x)
    @test_approx_eq v y

    out, y = zeros(4, 3), zeros(4)
    ForwardDiff.jacobian!(out, f!, y, x, yopts)
    @test_approx_eq out j
    @test_approx_eq y v

    out, y = zeros(4, 3), zeros(4)
    ForwardDiff.jacobian!(out, f!, y, x)
    @test_approx_eq out j
    @test_approx_eq y v

    out = JacobianResult(zeros(4), zeros(3))
    ForwardDiff.jacobian!(out, f!, x, Options(DiffBase.value(out), x))
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.jacobian(out) j

    out = JacobianResult(zeros(4), zeros(3))
    ForwardDiff.jacobian!(out, f!, x)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.jacobian(out) j
end

########################
# test vs. Calculus.jl #
########################

for f in DiffBase.ARRAY_TO_ARRAY_FUNCS
    v = f(X)
    j = ForwardDiff.jacobian(f, X)
    @test_approx_eq_eps j Calculus.jacobian(f, X, :forward) FINITEDIFF_ERROR
    for c in CHUNK_SIZES
        opts = Options{c}(X)

        println("  ...testing $f with chunk size = $c")
        out = ForwardDiff.jacobian(f, X, opts)
        @test_approx_eq out j

        out = similar(X, length(X), length(X))
        ForwardDiff.jacobian!(out, f, X, opts)
        @test_approx_eq out j
    end
end

for f! in DiffBase.INPLACE_ARRAY_TO_ARRAY_FUNCS
    v = zeros(Y)
    f!(v, X)
    j = ForwardDiff.jacobian(f!, zeros(Y), X)
    @test_approx_eq_eps j Calculus.jacobian(x -> (y = zeros(Y); f!(y, x); y), X, :forward) FINITEDIFF_ERROR
    for c in CHUNK_SIZES
        opts = Options{c}(X)
        yopts = Options{c}(zeros(Y), X)

        println("  ...testing $(f!) with chunk size = $c")
        y = zeros(Y)
        out = ForwardDiff.jacobian(f!, y, X, yopts)
        @test_approx_eq y v
        @test_approx_eq out j

        y = zeros(Y)
        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f!, y, X)
        @test_approx_eq y v
        @test_approx_eq out j

        out = JacobianResult(Y, X)
        ForwardDiff.jacobian!(out, f!, X)
        @test_approx_eq DiffBase.value(out) v
        @test_approx_eq DiffBase.jacobian(out) j
    end
end

end # module
