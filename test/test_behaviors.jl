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

#######################
# Promote type Issues #
#######################

# Test overloading of `promote_array_type` #
#------------------------------------------#

promtyp = Base.promote_array_type(Base.DotAddFun(),
                                  ForwardDiff.ForwardDiffNumber{2, Float64,
                                  Tuple{Float64, Float64}}, Float64)
fdiffnum = ForwardDiff.ForwardDiffNumber{2,Float64,Tuple{Float64,Float64}}
@test promtyp <: fdiffnum


promtyp = Base.promote_array_type(Base.DotAddFun(),
                                  ForwardDiff.GradientNumber{2, Float64,
                                  Tuple{Float64, Float64}}, Float64)
gradnum = ForwardDiff.GradientNumber{2,Float64,Tuple{Float64,Float64}}
@test promtyp <: gradnum

promtyp = Base.promote_array_type(Base.DotAddFun(),
                                  ForwardDiff.HessianNumber{2, Float64,
                                  Tuple{Float64, Float64}}, Float64)
hessnum = ForwardDiff.HessianNumber{2,Float64,Tuple{Float64,Float64}}
@test promtyp <: hessnum

promtyp = Base.promote_array_type(Base.DotAddFun(),
                                  ForwardDiff.TensorNumber{2, Float64,
                                  Tuple{Float64, Float64}}, Float64)
tensnum = ForwardDiff.TensorNumber{2,Float64,Tuple{Float64,Float64}}
@test promtyp <: tensnum


# Arithmetic element-wise functions #
#-----------------------------------#

N = 4
a = ones(N)
jac0 = reshape(vcat([[zeros(N*(i-1)); a; zeros(N^2-N*i)] for i = 1:N]...), N^2, N)

for op in (-, +, .-, .+, ./, .*)

    f = x -> [op(x[1], a); op(x[2], a); op(x[3], a); op(x[4], a)]

    # jacobian
    jac = ForwardDiff.jacobian(f, a)
    @test reduce(&, -jac + jac0 .== 0)

    f = x -> sum([op(x[1], a); op(x[2], a); op(x[3], a); op(x[4], a)])

    # hessian
    hess = ForwardDiff.hessian(f, a)
    @test reduce(&, -hess + zeros(N, N) .== 0)

    # tensor
    tens = ForwardDiff.tensor(f, a)
    @test reduce(&, -tens + zeros(N, N, N) .== 0)
end
