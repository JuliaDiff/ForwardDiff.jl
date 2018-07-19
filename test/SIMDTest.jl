module SIMDTest

using Test
using ForwardDiff: Dual, valtype
using InteractiveUtils: code_llvm

const DUALS = (Dual(1., 2., 3., 4.),
               Dual(1., 2., 3., 4., 5.),
               Dual(Dual(1., 2.), Dual(3., 4.)))


function simd_sum(x::Vector{T}) where T
    s = zero(T)
    @simd for i in eachindex(x)
        @inbounds s = s + x[i]
    end
    return s
end

for D in map(typeof, DUALS)
    plus_bitcode = sprint(io -> code_llvm(io, +, (D, D)))
    @test occursin("fadd <4 x double>", plus_bitcode)

    minus_bitcode = sprint(io -> code_llvm(io, -, (D, D)))
    @test occursin("fsub <4 x double>", minus_bitcode)

    times_bitcode = sprint(io -> code_llvm(io, *, (D, D)))
    @test occursin(r"fadd \<.*?x double\>", times_bitcode)
    @test occursin(r"fmul \<.*?x double\>", times_bitcode)

    div_bitcode = sprint(io -> code_llvm(io, /, (D, D)))
    @test occursin(r"fadd \<.*?x double\>", div_bitcode)
    @test occursin(r"fmul \<.*?x double\>", div_bitcode)

    exp_bitcode = sprint(io -> code_llvm(io, ^, (D, D)))
    @test occursin(r"fadd \<.*?x double\>", exp_bitcode)
    if !(valtype(D) <: Dual)
        # see https://github.com/JuliaDiff/ForwardDiff.jl/issues/167
        @test occursin(r"fmul \<.*?x double\>", exp_bitcode)

        # see https://github.com/JuliaDiff/ForwardDiff.jl/pull/201
        sum_bitcode = sprint(io -> code_llvm(io, simd_sum, (Vector{D},)))
        @test occursin(r"fadd (fast |)\<.*?x double\>", sum_bitcode)
    end
end

end # module
