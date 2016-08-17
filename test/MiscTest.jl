module MiscTest

import NaNMath

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

@test_approx_eq ForwardDiff.hessian(f, x) j(x)

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

@test_approx_eq tensor(rosenbrock, [0.1, 0.2, 0.3]) test_tensor_output

# Issue #59 example #
#-------------------#

x = rand(2)

f = x -> sin(x)/3 * cos(x)/2
df = x -> ForwardDiff.derivative(f, x)
testdf = x -> (((cos(x)^2)/3) - (sin(x)^2)/3) / 2
f2 = x -> df(x[1]) * f(x[2])
testf2 = x -> testdf(x[1]) * f(x[2])

@test_approx_eq ForwardDiff.gradient(f2, x) ForwardDiff.gradient(testf2, x)

######################################
# Higher-Dimensional Differentiation #
######################################

x = rand(3, 3)

@test_approx_eq ForwardDiff.jacobian(inv, x) -kron(inv(x'), inv(x))

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

for op in (-, +, .-, .+, ./, .*)
    f = x -> [op(x[1], a); op(x[2], a); op(x[3], a); op(x[4], a)]
    jac = ForwardDiff.jacobian(f, a)
    @test_approx_eq jac0 jac
end

# NaNs #
#------#

@test ForwardDiff.partials(NaNMath.pow(ForwardDiff.Dual(-2.0,1.0),ForwardDiff.Dual(2.0,0.0)),1) == -4.0

end
