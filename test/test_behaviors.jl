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

# Mixing chunk mode and vector mode #
#-----------------------------------#
x = rand(2*ForwardDiff.tuple_usage_threshold) # big enough to trigger vector mode

f = x -> sum(sin, x) + prod(tan, x) * sum(sqrt, x)
g = ForwardDiff.gradient(f) # gradient in vector mode
j = x -> ForwardDiff.jacobian(g, x, chunk_size=2)/2 # jacobian in chunk_mode

@test_approx_eq ForwardDiff.hessian(f, x) 2*j(x)

#####################
# Conversion Issues #
#####################

# Target function returns a literal (Issue #71) #
#-----------------------------------------------#

@test ForwardDiff.derivative(x->zero(x), rand()) == ForwardDiff.derivative(x->1.0, rand())
@test ForwardDiff.gradient(x->zero(x[1]), [rand()]) == ForwardDiff.gradient(x->1.0, [rand()])
@test ForwardDiff.hessian(x->zero(x[1]), [rand()]) == ForwardDiff.hessian(x->1.0, [rand()])
@test ForwardDiff.jacobian(x->[zero(x[1])], [rand()]) == ForwardDiff.jacobian(x->[1.0], [rand()])
