module SIMDTest

using Base.Test
using ForwardDiff: Dual

const DUALS = (Dual(1., 2., 3., 4.),
               Dual(1., 2., 3., 4., 5.),
               Dual(Dual(1., 2.), Dual(3., 4.)))


function simd_sum{T}(x::Vector{T})
    s = zero(T)
    @simd for i in eachindex(x)
        @inbounds s = s + x[i]
    end
    return s
end

for D in map(typeof, DUALS)
    plus_bitcode = sprint(io -> code_llvm(io, +, (D, D)))
    @test contains(plus_bitcode, "fadd <4 x double>")

    minus_bitcode = sprint(io -> code_llvm(io, -, (D, D)))
    @test contains(minus_bitcode, "fsub <4 x double>")

    times_bitcode = sprint(io -> code_llvm(io, *, (D, D)))
    @test ismatch(r"fadd \<.*?x double\>", times_bitcode)
    @test ismatch(r"fmul \<.*?x double\>", times_bitcode)

    div_bitcode = sprint(io -> code_llvm(io, /, (D, D)))
    @test ismatch(r"fadd \<.*?x double\>", div_bitcode)
    @test ismatch(r"fmul \<.*?x double\>", div_bitcode)

    exp_bitcode = sprint(io -> code_llvm(io, ^, (D, D)))
    @test ismatch(r"fadd \<.*?x double\>", exp_bitcode)
    @test ismatch(r"fmul \<.*?x double\>", exp_bitcode)

    sum_bitcode = sprint(io -> code_llvm(io, simd_sum, (Vector{D},)))
    @test ismatch(r"fadd \<.*?x double\>", sum_bitcode)
end

end # module
