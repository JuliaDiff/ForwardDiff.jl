module ConfusionTest

using Test
using ForwardDiff

using LinearAlgebra
using StaticArrays
using DiffResults

# Perturbation Confusion (Issue #83) #
#------------------------------------#

D = ForwardDiff.derivative

@test D(x -> x * D(y -> x + y, 1), 1) == 1
@test ForwardDiff.gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1]) == ForwardDiff.gradient(v -> sum(v) * norm(v), [1])



const A = rand(10,8)
y = rand(10)
x = rand(8)

@test A == ForwardDiff.jacobian(x) do x
    ForwardDiff.gradient(y) do y
        dot(y, A*x)
    end
end

# Issue #238                         #
#------------------------------------#

m,g = 1, 9.8
t = 1
q = [1,2]
q̇ = [3,4]
L(t,q,q̇) = m/2 * dot(q̇,q̇) - m*g*q[2]

∂L∂q̇(L, t, q, q̇) = ForwardDiff.gradient(a->L(t,q,a), q̇)
Dqq̇(L, t, q, q̇) = ForwardDiff.jacobian(a->∂L∂q̇(L,t,a,q̇), q)
@test Dqq̇(L, t, q, q̇)  == fill(0.0, 2, 2)


q = [1,2]
p = [5,6]
function Legendre_transformation(F, w)
    z = fill(0.0, size(w))
    M = ForwardDiff.hessian(F, z)
    b = ForwardDiff.gradient(F, z)
    v = cholesky(M)\(w-b)
    dot(w,v) - F(v)
end
function Lagrangian2Hamiltonian(Lagrangian, t, q, p)
    L = q̇ -> Lagrangian(t, q, q̇)
    Legendre_transformation(L, p)
end

Lagrangian2Hamiltonian(L, t, q, p)
@test ForwardDiff.gradient(a->Lagrangian2Hamiltonian(L, t, a, p), q) == [0.0,g]


#267: let scoping
@noinline f83a(z, x) = x[1]
z83a = ([(1, (2), [(3, (4, 5, [1, 2, (3, (4, 5), [5])]), (5))])])
let z = z83a
    g = x -> f83a(z, x)
    h = x -> g(x)
    @test ForwardDiff.hessian(h, [1.]) == zeros(1, 1)
end

@test ForwardDiff.derivative(1.0) do x
    ForwardDiff.derivative(x) do y
        x
    end
end == 0.0

# Nested derivatives whose tags are not ordered by containment
const captured = Ref{Any}()
const inner = Ref{Any}()
struct AFunction end
struct BFunction end
(::AFunction)(x) = captured[] * x^2
(::BFunction)(x) = captured[] * x^2
struct ADerivative end
struct BDerivative end
(::ADerivative)(x) = (captured[] = x; D(inner[], 2.0))
(::BDerivative)(x) = (captured[] = x; D(inner[], 2.0))
struct AGradient end
struct BGradient end
(::AGradient)(x) = (captured[] = prod(x); D(inner[], 2.0))
(::BGradient)(x) = (captured[] = prod(x); D(inner[], 2.0))
for (outer, f) in ((ADerivative(), BFunction()), (BDerivative(), AFunction()))
    inner[] = f
    @test D(outer, 3.0) == 4.0
end
for (outer, f) in ((AGradient(), BFunction()), (BGradient(), AFunction()))
    inner[] = f
    @test ForwardDiff.gradient(outer, [3.0, 2.0]) == [8.0, 12.0]
    @test ForwardDiff.hessian(outer, [3.0, 2.0]) == [0.0 4.0; 4.0 0.0]
end

# Every tag is greater than the tags in its input type and its function type
let T = ForwardDiff.Tag{AFunction,Float64}, d = ForwardDiff.Dual{T}(1.0, 1.0), f = x -> d * x
    for S in (ForwardDiff.Tag{BFunction,typeof(d)}, ForwardDiff.Tag{typeof(f),Float64})
        @test ForwardDiff.:≺(T, S)
        @test !ForwardDiff.:≺(S, T)
    end
end

# Nested `Dual`s store the greatest tag outermost
struct ATag end
struct BTag end
@test ForwardDiff.Dual{BTag}(ForwardDiff.Dual{ATag}(1.0, 2.0), ForwardDiff.Dual{ATag}(3.0, 4.0)) isa ForwardDiff.Dual{BTag}
@test_throws ArgumentError("Cannot store a Dual with tag $ATag outside a Dual with tag $BTag, since $ATag ≺ $BTag.") ForwardDiff.Dual{ATag}(ForwardDiff.Dual{BTag}(1.0, 2.0), ForwardDiff.Dual{BTag}(3.0, 4.0))
@test_throws ArgumentError("Cannot store a Dual with tag $ATag outside a Dual with tag $BTag, since $ATag ≺ $BTag.") ForwardDiff.Dual{ATag,Real,1}(ForwardDiff.Dual{BTag}(1.0, 2.0), ForwardDiff.Partials{1,Real}((3.0,)))
@test_throws ArgumentError("Cannot store a Dual with tag $ATag outside a Dual with tag $BTag, since $ATag ≺ $BTag.") ForwardDiff.Dual{ATag,Real,1}(1.0, ForwardDiff.Partials{1,Real}((ForwardDiff.Dual{BTag}(3.0, 4.0),)))

# Tags of nested `Dual`s are unique
@test_throws ArgumentError("Cannot store a Dual with tag $ATag inside a Dual with the same tag.") ForwardDiff.Dual{ATag}(ForwardDiff.Dual{ATag}(1.0, 2.0), ForwardDiff.Dual{ATag}(3.0, 4.0))
let d = ForwardDiff.Dual(ForwardDiff.Dual(1.0, 2.0), ForwardDiff.Dual(3.0, 4.0))
    T = ForwardDiff.Tag{Nothing,Float64}
    S = ForwardDiff.Tag{Nothing,ForwardDiff.Dual{T,Float64,1}}
    @test d isa ForwardDiff.Dual{S}
    @test ForwardDiff.value(T, ForwardDiff.value(S, d)) == 1.0
    @test ForwardDiff.partials(T, ForwardDiff.value(S, d), 1) == 2.0
    @test ForwardDiff.value(T, ForwardDiff.partials(S, d, 1)) == 3.0
    @test ForwardDiff.partials(T, ForwardDiff.partials(S, d, 1), 1) == 4.0
end
let f = x -> x[1]^2 * x[2], x = [3.0, 2.0]
    @test ForwardDiff.hessian(f, x, ForwardDiff.HessianConfig(nothing, x)) == [4.0 6.0; 6.0 0.0]
end

# Issue #845: all Hessian paths agree with the Jacobian of the gradient
strip_outer(x) = x
strip_outer(d::ForwardDiff.Dual{T}) where {T} = ForwardDiff.value(T, d)
f845a(z) = sum(abs2, z) + strip_outer(z[1]) * z[2]
f845b(z) = strip_outer(sum(abs2, z))
for f in (f845a, f845b), x in ([1.0, 2.0, 3.0], SVector(1.0, 2.0, 3.0))
    H = ForwardDiff.jacobian(y -> ForwardDiff.gradient(f, y), x)
    g = ForwardDiff.gradient(f, x)
    @test ForwardDiff.hessian(f, x) == H
    for c in 1:3
        @test ForwardDiff.hessian(f, x, ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}())) == H
    end
    out = fill(NaN, 3, 3)
    @test ForwardDiff.hessian!(out, f, x) === out
    @test out == H
    for result in (DiffResults.HessianResult(x), DiffResults.DiffResult(NaN, (fill(NaN, 3), fill(NaN, 3, 3))))
        r = ForwardDiff.hessian!(result, f, x)
        if result isa DiffResults.MutableDiffResult
            @test r === result
        end
        @test DiffResults.value(r) == f(x)
        @test DiffResults.gradient(r) == g
        @test DiffResults.hessian(r) == H
    end
end


end # module
