module MiscTest

import NaNMath

using Compat
using Base.Test
using ForwardDiff

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

@test isapprox(tensor(DiffBase.rosenbrock_1, [0.1, 0.2, 0.3]), test_tensor_output)

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

# Perturbation Confusion (Issue #83) #
#------------------------------------#

D = ForwardDiff.derivative

@test_throws ForwardDiff.TagMismatchError D(x -> x * D(y -> x + y, 1), 1)
@test_throws ForwardDiff.TagMismatchError ForwardDiff.gradient(v -> sum(v) * D(y -> y * norm(v), 1), [1])

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
@test isapprox(jinvx, ForwardDiff.jacobian(inv, Compat.view(x, 1:5, 1:5)))

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
a = ones(N)
jac0 = reshape(vcat([[zeros(N*(i-1)); a; zeros(N^2-N*i)] for i = 1:N]...), N^2, N)

for op in (:-, :+, :.-, :.+, :./, :.*)
    f = @eval x -> [$op(x[1], a); $op(x[2], a); $op(x[3], a); $op(x[4], a)]
    jac = @eval ForwardDiff.jacobian(f, a)
    @test isapprox(jac0, jac)
end

# NaNs #
#------#

@test ForwardDiff.partials(NaNMath.pow(ForwardDiff.Dual(-2.0,1.0),ForwardDiff.Dual(2.0,0.0)),1) == -4.0

end
