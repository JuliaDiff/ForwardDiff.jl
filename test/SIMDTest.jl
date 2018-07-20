module SIMDTest

using Test
using ForwardDiff: Dual, valtype
using InteractiveUtils: code_llvm
using StaticArrays: SVector

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

@testset "SIMD $D" for D in map(typeof, DUALS)
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

    pow_bitcode = sprint(io -> code_llvm(io, ^, (D, Int)))
    @test occursin(r"fmul \<.*?x double\>", pow_bitcode)

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

# `pow2dot` is chosen so that `@code_llvm pow2dot(SVector(1:1.0:4...))`
# generates code with SIMD instructions.
# See:
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/332
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/331#issuecomment-406107260
@inline pow2(x) = x^2
pow2dot(xs) = pow2.(xs)

# Nested dual such as `Dual(Dual(1., 2.), Dual(3., 4.))` only produces
# "fmul <2 x double>" so it is excluded from the following test.
const POW_DUALS = (Dual(1., 2.),
                   Dual(1., 2., 3.),
                   Dual(1., 2., 3., 4.),
                   Dual(1., 2., 3., 4., 5.))

@testset "SIMD square of $D" for D in map(typeof, POW_DUALS)
    pow_bitcode = sprint(io -> code_llvm(io, pow2dot, (SVector{4, D},)))
    @test occursin(r"(.*fmul \<4 x double\>){2}"s, pow_bitcode)
    # "{2}" is for asserting that fmul has to appear at least twice:
    # once for `.value` and once for `.partials`.
end

end # module
