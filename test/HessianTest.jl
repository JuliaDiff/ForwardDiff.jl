module HessianTest

import Calculus

using Test
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Tag
using StaticArrays
using DiffTests

include(joinpath(dirname(@__FILE__), "utils.jl"))

#############################
# rosenbrock hardcoded test #
#############################

f = DiffTests.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]
h = [-66.0  -40.0    0.0;
     -40.0  130.0  -80.0;
       0.0  -80.0  200.0]

@testset "running hardcoded test with chunk size = $c and tag = $(repr(tag))" for c in HESSIAN_CHUNK_SIZES, tag in (nothing, Tag((f,ForwardDiff.gradient), eltype(x)))
    cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}(), tag)
    resultcfg = ForwardDiff.HessianConfig(f, DiffResults.HessianResult(x), x, ForwardDiff.Chunk{c}(), tag)

    @test eltype(resultcfg) == eltype(cfg)

    @test isapprox(h, ForwardDiff.hessian(f, x))
    @test isapprox(h, ForwardDiff.hessian(f, x, cfg))

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x)
    @test isapprox(out, h)

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x, cfg)
    @test isapprox(out, h)

    out = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.gradient(out), g)
    @test isapprox(DiffResults.hessian(out), h)

    out = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(out, f, x, resultcfg)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.gradient(out), g)
    @test isapprox(DiffResults.hessian(out), h)
end

cfgx = ForwardDiff.HessianConfig(sin, x)
@test_throws ForwardDiff.InvalidTagException ForwardDiff.hessian(f, x, cfgx)
@test ForwardDiff.hessian(f, x, cfgx, Val{false}()) == ForwardDiff.hessian(f,x)


########################
# test vs. Calculus.jl #
########################

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    h = ForwardDiff.hessian(f, X)
    # finite difference approximation error is really bad for Hessians...
    @test isapprox(h, Calculus.hessian(f, X), atol=0.02)
    @testset "$f with chunk size = $c and tag = $(repr(tag))" for c in HESSIAN_CHUNK_SIZES, tag in (nothing, Tag((f,ForwardDiff.gradient), eltype(x)))
        cfg = ForwardDiff.HessianConfig(f, X, ForwardDiff.Chunk{c}(), tag)
        resultcfg = ForwardDiff.HessianConfig(f, DiffResults.HessianResult(X), X, ForwardDiff.Chunk{c}(), tag)

        out = ForwardDiff.hessian(f, X, cfg)
        @test isapprox(out, h)

        out = similar(X, length(X), length(X))
        ForwardDiff.hessian!(out, f, X, cfg)
        @test isapprox(out, h)

        out = DiffResults.HessianResult(X)
        ForwardDiff.hessian!(out, f, X, resultcfg)
        @test isapprox(DiffResults.value(out), v)
        @test isapprox(DiffResults.gradient(out), g)
        @test isapprox(DiffResults.hessian(out), h)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

@info "testing specialized StaticArray codepaths"

x = rand(3, 3)
for T in (StaticArrays.SArray, StaticArrays.MArray)
    sx = T{Tuple{3,3}}(x)

    cfg = ForwardDiff.HessianConfig(nothing, x)
    scfg = ForwardDiff.HessianConfig(nothing, sx)

    actual = ForwardDiff.hessian(prod, x)
    @test ForwardDiff.hessian(prod, sx) == actual
    @test ForwardDiff.hessian(prod, sx, cfg) == actual
    @test ForwardDiff.hessian(prod, sx, scfg) == actual
    @test ForwardDiff.hessian(prod, sx, scfg) isa StaticArray
    @test ForwardDiff.hessian(prod, sx, scfg, Val{false}()) == actual
    @test ForwardDiff.hessian(prod, sx, scfg, Val{false}()) isa StaticArray

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx)
    @test out == actual

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx, cfg)
    @test out == actual

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx, scfg)
    @test out == actual

    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, prod, x)

    result1 = DiffResults.HessianResult(x)
    result2 = DiffResults.HessianResult(x)
    result3 = DiffResults.HessianResult(x)
    result1 = ForwardDiff.hessian!(result1, prod, sx)
    result2 = ForwardDiff.hessian!(result2, prod, sx, ForwardDiff.HessianConfig(prod, result2, x, ForwardDiff.Chunk(x), nothing))
    result3 = ForwardDiff.hessian!(result3, prod, sx, ForwardDiff.HessianConfig(prod, result3, x, ForwardDiff.Chunk(x), nothing))
    @test DiffResults.value(result1) == DiffResults.value(result)
    @test DiffResults.value(result2) == DiffResults.value(result)
    @test DiffResults.value(result3) == DiffResults.value(result)
    @test DiffResults.gradient(result1) == DiffResults.gradient(result)
    @test DiffResults.gradient(result2) == DiffResults.gradient(result)
    @test DiffResults.gradient(result3) == DiffResults.gradient(result)
    @test DiffResults.hessian(result1) == DiffResults.hessian(result)
    @test DiffResults.hessian(result2) == DiffResults.hessian(result)
    @test DiffResults.hessian(result3) == DiffResults.hessian(result)

    sresult1 = DiffResults.HessianResult(sx)
    sresult2 = DiffResults.HessianResult(sx)
    sresult3 = DiffResults.HessianResult(sx)
    sresult1 = ForwardDiff.hessian!(sresult1, prod, sx)
    sresult2 = ForwardDiff.hessian!(sresult2, prod, sx, ForwardDiff.HessianConfig(prod, sresult2, x, ForwardDiff.Chunk(x), nothing))
    sresult3 = ForwardDiff.hessian!(sresult3, prod, sx, ForwardDiff.HessianConfig(prod, sresult3, x, ForwardDiff.Chunk(x), nothing))
    @test DiffResults.value(sresult1) == DiffResults.value(result)
    @test DiffResults.value(sresult2) == DiffResults.value(result)
    @test DiffResults.value(sresult3) == DiffResults.value(result)
    @test DiffResults.gradient(sresult1) == DiffResults.gradient(result)
    @test DiffResults.gradient(sresult2) == DiffResults.gradient(result)
    @test DiffResults.gradient(sresult3) == DiffResults.gradient(result)
    @test DiffResults.hessian(sresult1) == DiffResults.hessian(result)
    @test DiffResults.hessian(sresult2) == DiffResults.hessian(result)
    @test DiffResults.hessian(sresult3) == DiffResults.hessian(result)
end

@testset "branches in dot" begin
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/551
    H = [1 2 3; 4 5 6; 7 8 9];
    @test ForwardDiff.hessian(x->dot(x,H,x), fill(0.00001, 3)) ≈ [2 6 10; 6 10 14; 10 14 18]
    @test ForwardDiff.hessian(x->dot(x,H,x), zeros(3)) ≈ [2 6 10; 6 10 14; 10 14 18]
end

@testset "nested duals in general eigen" begin
    # The eigenvalue derivatives have to be computed from `value.(A)` alone; mixing that
    # level with `A` itself applies the product rule at the wrong level for nested `Dual`s
    B(w) = [3.0+w[1] 1.0+w[2]; 0.4+w[2] 2.0-2*w[1]]
    w = [0.11, -0.07]
    # sum(eigvals(B(w))) == tr(B(w)) == 5 - w[1] is linear in `w`
    @test ForwardDiff.hessian(w -> sum(eigvals(B(w))), w) ≈ zeros(2, 2) atol=1e-12
    # sum(eigvals(B(w)) .^ 2) == tr(B(w)^2) is quadratic in `w`
    @test ForwardDiff.hessian(w -> sum(eigvals(B(w)) .^ 2), w) ≈ [10 0; 0 4]

    # keyword arguments reach every level of the nesting
    @test ForwardDiff.hessian(w -> sum(eigvals(B(w); sortby = nothing)), w) ≈ zeros(2, 2) atol=1e-12
    @test ForwardDiff.hessian(w -> sum(eigvals(B(w); permute = false, scale = false) .^ 2), w) ≈ [10 0; 0 4]

    # complex eigenvalues: λ = w[1] ± im*(1 + w[2]), i.e. sum(abs2, λ) == 2*(w[1]^2 + (1 + w[2])^2)
    C(w) = [w[1] -1.0-w[2]; 1.0+w[2] w[1]]
    @test ForwardDiff.hessian(w -> sum(abs2, eigvals(C(w))), [0.3, 0.2]) ≈ [4 0; 0 4]

    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/111. `S(w)` is Hermitian in its
    # values *and* its partials, so both sides take the `Symmetric` shortcut: this pins the
    # dispatch, not the general path, and the two Hessians are bitwise identical.
    S(w) = [w[1]^2 w[1]*w[2]*w[3]; w[1]*w[2]*w[3] w[2]^2]
    g(w) = sum(log, eigvals(S(w)))
    gsym(w) = sum(log, eigvals(Symmetric(S(w))))
    w111 = [0.9, 1.4, 0.3]
    @test ishermitian(S(w111))
    @test ForwardDiff.hessian(g, w111) == ForwardDiff.hessian(gsym, w111)

    # The same log-determinant Hessian on a matrix that is *not* Hermitian, so that the
    # general path is what is under test. `det(Bgen(w)) == w[1]^2 * (1 + w[2]^2 / 2)`, which
    # gives a closed-form reference that never goes through `eigen`.
    Bgen(w) = [w[1]^2 w[1]*w[2]; 0.5*w[1]*w[2] w[2]^2+1]
    hgen(w) = sum(log, eigvals(Bgen(w)))
    wgen = [0.9, 1.4]
    @test !ishermitian(Bgen(wgen))
    @test isreal(eigvals(Bgen(wgen)))
    @test ForwardDiff.hessian(hgen, wgen) ≈
          [-2/wgen[1]^2 0; 0 (1 - wgen[2]^2/2)/(1 + wgen[2]^2/2)^2]
    @test ForwardDiff.hessian(hgen, wgen) ≈ ForwardDiff.hessian(w -> log(det(Bgen(w))), wgen)

    # eigenvectors
    A0 = [2.0 1.0 0.5; 0.5 3.0 1.5; 0.25 0.75 4.0]
    function v1(x)
        v = eigen(reshape(x, 3, 3)).vectors[:, 1]
        return sum(abs2, v .- [1.0, 0.5, -0.2])
    end
    x0 = vec(A0)
    @test ForwardDiff.hessian(v1, x0) ≈ ForwardDiff.jacobian(x -> ForwardDiff.gradient(v1, x), x0)
    @test ForwardDiff.hessian(v1, x0) ≈ Calculus.finite_difference_jacobian(x -> ForwardDiff.gradient(v1, x), x0) atol=1e-5
end

# Helpers for the eigenvector phase gauge tests below
struct GaugeTag end

# A deterministic `n x n` matrix with a complex spectrum and well separated eigenvalues
function complex_spectrum_matrix(n)
    A = zeros(n, n)
    for b in 1:(n ÷ 2)
        i = 2b - 1
        A[i, i] = A[i+1, i+1] = 0.5 + b / 4
        A[i, i+1] = -1.0 - b / 8
        A[i+1, i] = 1.0 + b / 8
    end
    isodd(n) && (A[n, n] = 2.0)
    for i in 1:n, j in 1:n
        A[i, j] += 0.15 * sinpi((i + 2j) / (n + 1))
    end
    return A
end

seed_matrix(n, i, j) = (S = zeros(n, n); S[i, j] = 1.0; S)

# `A` lifted to a `Dual` of nesting depth `k` with vanishing partials
lift_dual(A, k) = k == 0 ? A : Dual{GaugeTag}.(lift_dual(A, k - 1), lift_dual(zero(A), k - 1))
# `A` seeded with one partial per level, innermost seed first, the way `hessian` nests them
function nest_dual(A, seeds...)
    M = A
    for (k, seed) in enumerate(seeds)
        M = Dual{GaugeTag}.(M, lift_dual(seed, k - 1))
    end
    return M
end

# The entry `eigen` made real, i.e. the one of largest magnitude
phase_index(v) = argmax(j -> abs2(v[j]), eachindex(v))

# All components of a (possibly nested) `Dual`: its value and every partial, recursively
function dual_components!(out, x)
    if x isa Dual
        dual_components!(out, ForwardDiff.value(x))
        for i in 1:ForwardDiff.npartials(x)
            dual_components!(out, ForwardDiff.partials(x, i))
        end
    else
        push!(out, x)
    end
    return out
end
dual_components(x) = dual_components!(Float64[], x)

# The eigenvectors and their first derivative in direction `seed`, with the columns flipped
# to match the signs of `Vref`
function eigvecs_and_derivative(X, seed, Vref)
    V = eigen(Dual{GaugeTag}.(X, seed)).vectors
    V0 = map(z -> complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z))), V)
    V1 = map(z -> complex(ForwardDiff.partials(real(z), 1), ForwardDiff.partials(imag(z), 1)), V)
    for i in axes(V0, 2)
        k = phase_index(view(Vref, :, i))
        if real(V0[k, i]) * real(Vref[k, i]) < 0
            V0[:, i] .*= -1
            V1[:, i] .*= -1
        end
    end
    return V0, V1
end

@testset "eigenvector phase gauge under nesting, n = $n" for n in 2:6
    A = complex_spectrum_matrix(n)
    # otherwise the phase convention never comes up
    @test !isreal(eigvals(A))
    seeds = (seed_matrix(n, 1, 1), seed_matrix(n, 2, min(3, n)), seed_matrix(n, min(3, n), 1))

    # `eigen` returns eigenvectors of unit 2-norm whose largest-magnitude entry is real, so
    # `imag(u[k]) == 0` and `u' * u == 1` hold identically along the curve and every
    # derivative of them has to vanish. A rounding residue in the first is invisible at
    # first order, but one level up it sits in the partials, where `isreal` sees it, and
    # `_findrealmaxabs2` then fixes the phase at the wrong entry.
    @testset "nesting depth $depth" for depth in 1:3
        V = eigen(nest_dual(A, seeds[1:depth]...)).vectors
        for i in axes(V, 2)
            v = view(V, :, i)
            @test iszero(imag(V[phase_index(v), i]))
            @test all(x -> abs(x) < 1e-12, dual_components(real(dot(v, v)) - 1))
        end
    end

    # second derivatives of the eigenvectors, against central differences of the first
    # derivatives. `eigen` does not pin the sign of the real entry, so the columns have to
    # be realigned before differencing; plain central differences are off by `O(1)`.
    Ea, Eb = seeds[1], seeds[2]
    Vref = eigen(A).vectors
    V = eigen(nest_dual(A, Ea, Eb)).vectors
    ad = map(z -> complex(ForwardDiff.partials(ForwardDiff.partials(real(z), 1), 1),
                          ForwardDiff.partials(ForwardDiff.partials(imag(z), 1), 1)), V)
    h = 1e-5
    _, Dp = eigvecs_and_derivative(A .+ h .* Eb, Ea, Vref)
    _, Dm = eigvecs_and_derivative(A .- h .* Eb, Ea, Vref)
    @test maximum(abs, ad .- (Dp .- Dm) ./ (2h)) < 1e-6
end

#https://github.com/JuliaDiff/ForwardDiff.jl/issues/720
@testset "allocation-free hessian with StaticArrays" begin
    function hessian_allocs()
        g = r -> (r[1]^2 - 3) * (r[2]^2 - 2)
        x = SVector(0.5, 2.8)
        hres = DiffResults.HessianResult(x)
        return @allocated(ForwardDiff.hessian!(hres, g, x))
    end
    @test iszero(hessian_allocs())
end

end # module
