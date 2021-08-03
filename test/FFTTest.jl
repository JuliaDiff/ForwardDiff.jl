module FFTTest

using Test
using ForwardDiff: Dual, valtype, value, partials, derivative
using FFTW
using AbstractFFTs: complexfloat, realfloat


x1 = Dual.(1:4.0, 2:5, 3:6)

@test value.(x1) == 1:4
@test partials.(x1, 1) == 2:5

@test complexfloat(x1)[1] === complexfloat(x1[1]) === Dual(1.0, 2.0, 3.0) + 0im
@test realfloat(x1)[1] === realfloat(x1[1]) === Dual(1.0, 2.0, 3.0)

@test fft(x1, 1)[1] isa Complex{<:Dual}

@testset "$f" for f in [fft, ifft, rfft, bfft]
    @test value.(f(x1)) == f(value.(x1))
    @test partials.(f(x1), 1) == f(partials.(x1, 1))
end

f = x -> real(fft([x; 0; 0])[1])
@test derivative(f,0.1) ≈ 1

r = x -> real(rfft([x; 0; 0])[1])
@test derivative(r,0.1) ≈ 1

# c = x -> dct([x; 0; 0])[1]
# @test derivative(c,0.1) ≈ 1
end # module
