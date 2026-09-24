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


end # module
