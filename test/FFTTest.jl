module FFTTest

using FFTW, LinearAlgebra, Test
using ForwardDiff: Dual, valtype, value, partials, derivative
using AbstractFFTs: complexfloat, realfloat

@testset "fft and rfft" begin
    x1 = Dual.(1:4.0, 2:5, 3:6)

    @test value.(x1) == 1:4
    @test partials.(x1, 1) == 2:5
    @test partials.(x1, 2) == 3:6

    @test complexfloat(x1)[1] === complexfloat(x1[1]) === Dual(1.0, 2.0, 3.0) + 0im
    @test realfloat(x1)[1] === realfloat(x1[1]) === Dual(1.0, 2.0, 3.0)

    @test fft(x1, 1)[1] isa Complex{<:Dual}

    @testset "$f" for f in (fft, ifft, rfft, bfft)
        @test value.(f(x1)) == f(value.(x1))
        @test partials.(f(x1), 1) == f(partials.(x1, 1))
        @test partials.(f(x1), 2) == f(partials.(x1, 2))
    end

    @test ifft(fft(x1)) == x1
    @test irfft(rfft(x1), length(x1)) ≈ x1
    @test brfft(rfft(x1), length(x1)) ≈ 4x1

    f = x -> real(fft([x; 0; 0])[1])
    @test derivative(f,0.1) ≈ 1

    r = x -> real(rfft([x; 0; 0])[1])
    @test derivative(r,0.1) ≈ 1


    n = 100
    θ = range(0,2π; length=n+1)[1:end-1]
    # emperical from Mathematical
    @test derivative(ω -> fft(exp.(ω .* cos.(θ)))[1]/n, 1) ≈ 0.565159103992485

    # c = x -> dct([x; 0; 0])[1]
    # @test derivative(c,0.1) ≈ 1

    @testset "matrix" begin
        A = x1 * (1:10)'
        @test value.(fft(A)) == fft(value.(A))
        @test partials.(fft(A), 1) == fft(partials.(A, 1))
        @test partials.(fft(A), 2) == fft(partials.(A, 2))

        @test value.(fft(A, 1)) == fft(value.(A), 1)
        @test partials.(fft(A, 1), 1) == fft(partials.(A, 1), 1)
        @test partials.(fft(A, 1), 2) == fft(partials.(A, 2), 1)

        @test value.(fft(A, 2)) == fft(value.(A), 2)
        @test partials.(fft(A, 2), 1) == fft(partials.(A, 1), 2)
        @test partials.(fft(A, 2), 2) == fft(partials.(A, 2), 2)
    end

    c1 = complex.(x1)
    @test mul!(similar(c1), FFTW.plan_fft(x1), x1) == fft(x1)
    @test mul!(similar(c1), FFTW.plan_fft(c1), c1) == fft(c1)
end

@testset "r2r" begin
    x1 = Dual.(1:4.0, 2:5, 3:6)
    t = FFTW.r2r(x1, FFTW.R2HC)

    @test value.(t) == FFTW.r2r(value.(x1), FFTW.R2HC)
    @test partials.(t, 1) == FFTW.r2r(partials.(x1, 1), FFTW.R2HC)
    @test partials.(t, 2) == FFTW.r2r(partials.(x1, 2), FFTW.R2HC)

    t = FFTW.r2r(x1 + 2im*x1, FFTW.R2HC)
    @test value.(t) == FFTW.r2r(value.(x1 + 2im*x1), FFTW.R2HC)
    @test partials.(t, 1) == FFTW.r2r(partials.(x1 + 2im*x1, 1), FFTW.R2HC)
    @test partials.(t, 2) == FFTW.r2r(partials.(x1 + 2im*x1, 2), FFTW.R2HC)

    f = ω -> FFTW.r2r([ω; zeros(9)], FFTW.R2HC)[1]
    @test derivative(f, 0.1) ≡ 1.0

    @test mul!(similar(x1), FFTW.plan_r2r(x1, FFTW.R2HC), x1) == FFTW.r2r(x1, FFTW.R2HC)
end
end # module