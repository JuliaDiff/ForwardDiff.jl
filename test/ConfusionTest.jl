module ConfusionTest

using Test
using ForwardDiff

using LinearAlgebra

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

# Nested differentiation must not depend on tagcount instantiation order (#714) #
#-------------------------------------------------------------------------------#

# containstag: nesting through the seeded value type V
struct TagOrderOuterMarker end
struct TagOrderInnerMarker end
let Tag = ForwardDiff.Tag, Dual = ForwardDiff.Dual
    Touter = Tag{TagOrderOuterMarker, Float64}
    Tinner = Tag{TagOrderInnerMarker, Dual{Touter, Float64, 1}}
    @test ForwardDiff.containstag(Tinner, Touter)
    @test !ForwardDiff.containstag(Touter, Tinner)
    # bake tagcounts in inverted order: containment must win regardless
    ForwardDiff.tagcount(Tinner)
    ForwardDiff.tagcount(Touter)
    @test ForwardDiff.:(≺)(Touter, Tinner)
    @test !ForwardDiff.:(≺)(Tinner, Touter)
end

# Second derivative with tagcount baked in inverted order, as precompilation can
# do: the inner tag nests the outer through V, so ordering must not consult
# tagcount at all.
struct TagOrderInnerV end
(::TagOrderInnerV)(y) = y^3
struct TagOrderOuterV end
(::TagOrderOuterV)(x) = ForwardDiff.derivative(TagOrderInnerV(), x)
ForwardDiff.tagcount(ForwardDiff.Tag{TagOrderInnerV, ForwardDiff.Dual{ForwardDiff.Tag{TagOrderOuterV, Float64}, Float64, 1}})
ForwardDiff.tagcount(ForwardDiff.Tag{TagOrderOuterV, Float64})
@test ForwardDiff.derivative(TagOrderOuterV(), 2.0) ≈ 12.0

# Same with the outer perturbation entering through a capture in F (both tags
# have V === Float64): nesting is only visible through the callable's fields.
struct TagOrderOuterF end
struct TagOrderInnerF
    x_dual::ForwardDiff.Dual{ForwardDiff.Tag{TagOrderOuterF, Float64}, Float64, 1}
end
(c::TagOrderInnerF)(y) = sin(c.x_dual * y)
(::TagOrderOuterF)(x_dual::ForwardDiff.Dual{ForwardDiff.Tag{TagOrderOuterF, Float64}, Float64, 1}) =
    ForwardDiff.derivative(TagOrderInnerF(x_dual), 1.0)
ForwardDiff.tagcount(ForwardDiff.Tag{TagOrderInnerF, Float64})
ForwardDiff.tagcount(ForwardDiff.Tag{TagOrderOuterF, Float64})
@test ForwardDiff.derivative(TagOrderOuterF(), 0.5) ≈ cos(0.5) - 0.5 * sin(0.5)

# Three-level nesting where the innermost derivative is seeded with a plain
# Float64 while the outer perturbations enter through closure captures. A
# depth-only fast path mis-orders this case; it must keep working.
let
    inner_deriv(d) = ForwardDiff.derivative(y -> y^2 * d, 1.0)
    middle_grad(v) = ForwardDiff.gradient(u -> sum(inner_deriv(ui) * ui for ui in u), v)
    outer_fn(x) = sum(middle_grad([x, 2x]))
    @test ForwardDiff.derivative(outer_fn, 0.5) ≈ 12.0
end

end # module
