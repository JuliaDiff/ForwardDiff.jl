module GPUArraysTest

using ForwardDiff, Test
using JLArrays

# JLArrays emulates GPU array semantics (including the scalar-indexing ban) on
# the CPU, so the dense-array `seed!` fast paths can be exercised on a GPU-like
# array type without a physical GPU. GPU arrays are covered by the dense fast
# paths because `AbstractGPUArray <: DenseArray`.
JLArrays.allowscalar(false)

@testset "ForwardDiff seeding on GPU arrays" begin
    f(x) = x .^ 2 .+ 2 .* x

    @testset "jacobian, vector mode (length $n)" for n in (1, 4, 8)
        x = collect(Float64, 1:n)
        @test Array(ForwardDiff.jacobian(f, JLArray(x))) == ForwardDiff.jacobian(f, x)
    end

    # lengths above the chunk size exercise the chunked `seed!` methods
    @testset "jacobian, chunk mode (length $n, chunk $c)" for n in (16, 20, 27), c in (4, 8)
        x = collect(Float64, 1:n)
        cfg = ForwardDiff.JacobianConfig(f, JLArray(x), ForwardDiff.Chunk{c}())
        @test Array(ForwardDiff.jacobian(f, JLArray(x), cfg)) == ForwardDiff.jacobian(f, x)
    end

    @testset "jacobian! into a GPU array (length $n)" for n in (4, 16)
        x = collect(Float64, 1:n)
        out = JLArray(zeros(n, n))
        ForwardDiff.jacobian!(out, f, JLArray(x))
        @test Array(out) == ForwardDiff.jacobian(f, x)
    end

    @testset "jacobian of f! with GPU input and output" begin
        f!(y, x) = (y .= x .^ 2 .+ 2 .* x; nothing)
        x = collect(Float64, 1:8)
        y = zeros(8)
        J = ForwardDiff.jacobian(f!, JLArray(y), JLArray(x))
        @test Array(J) == ForwardDiff.jacobian(f!, y, x)
    end

    @testset "jacobian with matrix input (chunk $c)" for c in (3, 6)
        X = reshape(collect(Float64, 1:12), 4, 3)
        g(x) = x .* sum(x)
        cfg = ForwardDiff.JacobianConfig(g, JLArray(X), ForwardDiff.Chunk{c}())
        @test Array(ForwardDiff.jacobian(g, JLArray(X), cfg)) ≈ ForwardDiff.jacobian(g, X)
    end

    @testset "jacobian with view input" begin
        X = JLArray(reshape(collect(Float64, 1:18), 6, 3))
        xv = view(X, :, 2)
        @test Array(ForwardDiff.jacobian(f, xv)) == ForwardDiff.jacobian(f, Array(xv))
    end
end

end # module
