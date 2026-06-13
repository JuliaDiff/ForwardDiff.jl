module GPUArraysCoreTest

using ForwardDiff, Test
using JLArrays

# JLArrays emulates GPU array semantics (including the scalar-indexing ban) on
# the CPU, so the GPUArraysCore extension's broadcast-based `seed!` methods can
# be exercised without a physical GPU.
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
end

end # module
