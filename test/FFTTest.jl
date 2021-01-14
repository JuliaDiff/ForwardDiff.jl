module FFTTest

using Test
using ForwardDiff: Dual, valtype, value, partials
using FFTW
using AbstractFFTs: complexfloat, realfloat


x1 = Dual.(1:4.0, 2:5, 3:6)

@test value.(x1) == 1:4
@test partials.(x1, 1) == 2:5

@test complexfloat(x1)[1] === complexfloat(x1[1]) === Dual(1.0, 2.0, 3.0) + 0im
@test realfloat(x1)[1] === realfloat(x1[1]) === Dual(1.0, 2.0, 3.0)

@test fft(x1, 1)[1] isa Complex{<:Dual}

@testset "$f" for f in [fft, ifft, rfft, bfft]
    @test value.(fft(x1)) == fft(value.(x1))
    @test partials.(fft(x1), 1) == fft(partials.(x1, 1))
end



end # module
