module DerivativeTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

srand(1)

########################
# test vs. Calculus.jl #
########################

const x = 1

for f in DiffBase.NUMBER_TO_NUMBER_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = DiffBase.DiffResult(zero(v), zero(v))
    ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.derivative(out), d)
end

for f in DiffBase.NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)

    @test !(eltype(d) <: ForwardDiff.Dual)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = similar(v)
    ForwardDiff.derivative!(out, f, x)
    @test isapprox(out, d)

    out = DiffBase.DiffResult(similar(v), similar(d))
    ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.derivative(out), d)
end

##################
# n-ary versions #
##################

# (::Real, ::Real) -> ::Real #
#----------------------------#

f(a, b) = sin(a) * tan(b)

a, b = rand(2)

valf = f(a, b)
∇f = ForwardDiff.gradient(x -> f(x...), [a, b])

@test collect(ForwardDiff.derivative(f, (a, b))) == ∇f

out = (DiffBase.DiffResult(zero(a), zero(a)), DiffBase.DiffResult(zero(b), zero(b)))
ForwardDiff.derivative!(out, f, (a, b))
@test DiffBase.value(out[1]) == DiffBase.value(out[2]) == valf
@test [DiffBase.derivative(out[1]), DiffBase.derivative(out[2])] == ∇f

out = (Base.RefValue(zero(a)), DiffBase.DiffResult(zero(b), zero(b)))
ForwardDiff.derivative!(out, f, (a, b))
@test DiffBase.value(out[2]) == valf
@test [out[1][], DiffBase.derivative(out[2])] == ∇f

out = (DiffBase.DiffResult(zero(a), zero(a)), [zero(b)])
ForwardDiff.derivative!(out, f, (a, b))
@test DiffBase.value(out[1]) == valf
@test [DiffBase.derivative(out[1]), out[2][]] == ∇f

out = (Base.RefValue(zero(a)), [zero(b)])
ForwardDiff.derivative!(out, f, (a, b))
@test [out[1][], out[2][]] == ∇f

# (::Real, ::Real) -> ::Vector #
#------------------------------#

g(a, b) = cos.([f(a, b), f(b, a)]) .+ b .- a

a, b = rand(2)

valg = g(a, b)
Jg = ForwardDiff.jacobian(x -> g(x...), [a, b])

@test hcat(ForwardDiff.derivative(g, (a, b))...) == Jg

out = (DiffBase.DiffResult(similar(valg), similar(valg)), DiffBase.DiffResult(similar(valg), similar(valg)))
ForwardDiff.derivative!(out, g, (a, b))
@test DiffBase.value(out[1]) == DiffBase.value(out[2]) == valg
@test hcat(DiffBase.derivative(out[1]), DiffBase.derivative(out[2])) == Jg

out = (similar(valg), DiffBase.DiffResult(similar(valg), similar(valg)))
ForwardDiff.derivative!(out, g, (a, b))
@test DiffBase.value(out[2]) == valg
@test hcat(out[1], DiffBase.derivative(out[2])) == Jg

out = (DiffBase.DiffResult(similar(valg), similar(valg)), similar(valg))
ForwardDiff.derivative!(out, g, (a, b))
@test DiffBase.value(out[1]) == valg
@test hcat(DiffBase.derivative(out[1]), out[2]) == Jg

out = (similar(valg), similar(valg))
ForwardDiff.derivative!(out, g, (a, b))
@test hcat(out[1], out[2]) == Jg

end # module
