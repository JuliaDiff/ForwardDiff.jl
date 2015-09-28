using Base.Test
using ForwardDiff

##########################
# Nested Differentiation #
##########################

# README example #
#----------------#
x = rand(5)

f = x -> sum(sin, x) + prod(tan, x) * sum(sqrt, x)
g = ForwardDiff.gradient(f)
j = ForwardDiff.jacobian(g)

@test_approx_eq ForwardDiff.hessian(f, x) j(x)

# Issue #59 example #
#-------------------#
x = rand(2)

f = x -> sin(x)/3 * cos(x)/2
df = ForwardDiff.derivative(f)
testdf = x -> (((cos(x)^2)/3) - (sin(x)^2)/3) / 2
f2 = x -> df(x[1]) * f(x[2])
testf2 = x -> testdf(x[1]) * f(x[2])

@test_approx_eq ForwardDiff.gradient(f2, x) ForwardDiff.gradient(testf2, x)
