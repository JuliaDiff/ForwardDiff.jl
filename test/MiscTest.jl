module MiscTest

import NaNMath

using Compat
using Compat.Test
using ForwardDiff
using DiffTests
using StaticArrays

include(joinpath(dirname(@__FILE__), "utils.jl"))

##########################
# Nested Differentiation #
##########################

# README example #
#----------------#

x = rand(5)

f = x -> sum(sin, x) + prod(tan, x) * sum(sqrt, x)
g = x -> ForwardDiff.gradient(f, x)
j = x -> ForwardDiff.jacobian(g, x)

@test isapprox(ForwardDiff.hessian(f, x), j(x))

# higher-order derivatives #
#--------------------------#

function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

test_tensor_output = reshape([240.0  -400.0     0.0;
                             -400.0     0.0     0.0;
                                0.0     0.0     0.0;
                             -400.0     0.0     0.0;
                                0.0   480.0  -400.0;
                                0.0  -400.0     0.0;
                                0.0     0.0     0.0;
                                0.0  -400.0     0.0;
                                0.0     0.0     0.0], 3, 3, 3)

@test isapprox(tensor(DiffTests.rosenbrock_1, [0.1, 0.2, 0.3]), test_tensor_output)

test_nested_jacobian_output = [-sin(1)  0.0     0.0;
                               -0.0    -0.0    -0.0;
                               -0.0    -0.0    -0.0;
                                0.0     0.0     0.0;
                               -0.0    -sin(2) -0.0;
                               -0.0    -0.0    -0.0;
                                0.0     0.0     0.0;
                               -0.0    -0.0    -0.0;
                               -0.0    -0.0    -sin(3)]

sin_jacobian = x -> ForwardDiff.jacobian(y -> broadcast(sin, y), x)

@test isapprox(ForwardDiff.jacobian(sin_jacobian, [1., 2., 3.]), test_nested_jacobian_output)

# Issue #59 example #
#-------------------#

x = rand(2)

f = x -> sin(x)/3 * cos(x)/2
df = x -> ForwardDiff.derivative(f, x)
testdf = x -> (((cos(x)^2)/3) - (sin(x)^2)/3) / 2
f2 = x -> df(x[1]) * f(x[2])
testf2 = x -> testdf(x[1]) * f(x[2])

@test isapprox(ForwardDiff.gradient(f2, x), ForwardDiff.gradient(testf2, x))

######################################
# Higher-Dimensional Differentiation #
######################################

x = rand(5, 5)

@test isapprox(ForwardDiff.jacobian(inv, x), -kron(inv(x'), inv(x)))

#########################################
# Differentiation with non-Array inputs #
#########################################

x = rand(5,5)

# Sparse
f = x -> sum(sin, x) + prod(tan, x) * sum(sqrt, x)
gfx = ForwardDiff.gradient(f, x)
@test isapprox(gfx, ForwardDiff.gradient(f, sparse(x)))

# Views
jinvx = ForwardDiff.jacobian(inv, x)
@test isapprox(jinvx, ForwardDiff.jacobian(inv, view(x, 1:5, 1:5)))

########################
# Conversion/Promotion #
########################

# target function returns a literal (Issue #71) #
#-----------------------------------------------#

@test ForwardDiff.derivative(x->zero(x), rand()) == ForwardDiff.derivative(x->1.0, rand())
@test ForwardDiff.gradient(x->zero(x[1]), [rand()]) == ForwardDiff.gradient(x->1.0, [rand()])
@test ForwardDiff.hessian(x->zero(x[1]), [rand()]) == ForwardDiff.hessian(x->1.0, [rand()])
@test ForwardDiff.jacobian(x->[zero(x[1])], [rand()]) == ForwardDiff.jacobian(x->[1.0], [rand()])

# arithmetic element-wise functions #
#-----------------------------------#

N = 4
a = fill(1.0, N)
jac0 = reshape(vcat([[fill(0.0, N*(i-1)); a; fill(0.0, N^2-N*i)] for i = 1:N]...), N^2, N)

for op in (:-, :+, :.-, :.+, :./, :.*)
    f = @eval x -> [$op(x[1], a); $op(x[2], a); $op(x[3], a); $op(x[4], a)]
    jac = @eval ForwardDiff.jacobian(f, a)
    @test isapprox(jac0, jac)
end

# NaNs #
#------#

@test ForwardDiff.partials(NaNMath.pow(ForwardDiff.Dual(-2.0,1.0),ForwardDiff.Dual(2.0,0.0)),1) == -4.0

# Partials{0} #
#-------------#

x, y = rand(3), rand(3)
h = ForwardDiff.hessian(y -> sum(hypot.(x, y)), y)
@test h[1, 1] ≈ (x[1]^2) / (x[1]^2 + y[1]^2)^(3/2)
@test h[2, 2] ≈ (x[2]^2) / (x[2]^2 + y[2]^2)^(3/2)
@test h[3, 3] ≈ (x[3]^2) / (x[3]^2 + y[3]^2)^(3/2)
for i in 1:3, j in 1:3
    i != j && (@test h[i, j] ≈ 0.0)
end

################
# FieldVectors #
################

# issue 305

# Test mutable FieldVector

mutable struct MPoint{R<:Real} <: FieldVector{2,R}
    x::R
    y::R
end

StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(2,)}) where {P<:MPoint, R<:Real} = MPoint{R}

f305scalar(x::MPoint) = sum(x+x)
@test ForwardDiff.gradient(f305scalar, MPoint(1.0, 2.0)) == MPoint(2.0, 2.0)
@test ForwardDiff.hessian(f305scalar, MPoint(1.0, 2.0)) == zeros(2,2)

f305vector(x::MPoint) = x+x
@test ForwardDiff.jacobian(f305vector, MPoint(1.0, 2.0)) == [2.0 0.0; 0.0 2.0]

# Test correct dispatch in differentiation

struct DifferentPoint{R<:Real} <: FieldVector{2,R}
    x::R
    y::R
end

StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(2,)}) where {P<:DifferentPoint, R<:Real} = DifferentPoint{R}

f305dispatch(p) = sum(p.^2)
f305dispatch(p::MPoint) = p.x^2
f305dispatch(p::DifferentPoint) = p.y^2

@test ForwardDiff.gradient(f305dispatch, [1.0, 2.0]) == [2.0, 4.0]
@test ForwardDiff.gradient(f305dispatch, MPoint(1.0, 2.0)) == [2.0, 0.0]
@test ForwardDiff.gradient(f305dispatch, DifferentPoint(1.0, 2.0)) == [0.0, 4.0]

@test ForwardDiff.hessian(f305dispatch, [1.0, 2.0]) == 2.0*eye(2)
@test ForwardDiff.hessian(f305dispatch, MPoint(1.0, 2.0)) == diagm([2.0, 0.0]) # [2.0 0.0; 0.0 0.0]
@test ForwardDiff.hessian(f305dispatch, DifferentPoint(1.0, 2.0)) == diagm([0.0, 2.0]) # [0.0 0.0; 0.0 2.0]

f305dispatchvec(p) = p.^2
f305dispatchvec(p::MPoint) = MPoint(p.x^2, 1.0)
f305dispatchvec(p::DifferentPoint) = DifferentPoint(p.y^2, 1.0)

@test ForwardDiff.jacobian(f305dispatchvec, [1.0, 2.0]) == diagm([2.0, 4.0])
@test ForwardDiff.jacobian(f305dispatchvec, MPoint(1.0, 2.0)) == diagm([2.0, 0.0])
@test ForwardDiff.jacobian(f305dispatchvec, DifferentPoint(1.0, 2.0)) == [0.0 4.0; 0.0 0.0]


########
# misc #
########

# issue 267
@noinline f267(z, x) = x[1]
z267 = ([(1, (2), [(3, (4, 5, [1, 2, (3, (4, 5), [5])]), (5))])])
let z = z267
    g = x -> f267(z, x)
    h = x -> g(x)
    @test ForwardDiff.hessian(h, [1.]) == fill(0.0, 1, 1)
end

# issue #290
@test ForwardDiff.derivative(x -> rem2pi(x, RoundUp), rand()) == 1
@test ForwardDiff.derivative(x -> rem2pi(x, RoundDown), rand()) == 1

end # module
