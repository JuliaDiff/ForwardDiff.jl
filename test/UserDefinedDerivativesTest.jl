module UserDefinedDerivativesTest

##############
# Derivative #
##############

module Derivative

using Base.Test
using ForwardDiff

global d_counter = 0
f(x) = 3*x
Df(x) = (global d_counter += 1; return 3*one(x))
@ForwardDiff.implement_derivative f Df

@test ForwardDiff.derivative(f, rand()) == 3.0
@test d_counter == 1

# In function
function closure(a)
    d_counter = 0
    Dff(x) = (d_counter += 1; return a*one(x))
    ff(x) = a*x
    @ForwardDiff.implement_derivative ff Dff
    @test ForwardDiff.derivative(ff, rand()) == a
    @test d_counter == 1
end

closure(2.0)

end


############
# Gradient #
############

module Gradient

using Base.Test
using ForwardDiff

global g_counter = 0
f(x) = norm(x)
Df!(g, x) = (global g_counter += 1; copy!(g, x); scale!(g, 1 / norm(x)))
x = rand(5)
cfig = ForwardDiff.GradientImplementConfig(5, x)
ForwardDiff.@implement_gradient! f Df! cfig
@test ForwardDiff.gradient(f, x) ≈ x / norm(x)
@test g_counter == 1

# ForwardDiff.Chunk mode + preallocated config for user gradient
g_counter = 0
f2(x) = norm(x)
Df2(x) = (global g_counter += 1; x / norm(x))
x = rand(5)
ForwardDiff.@implement_gradient f2 Df2
cfg = ForwardDiff.GradientConfig(nothing, x, ForwardDiff.Chunk{2}())
@test ForwardDiff.gradient(f2, x, cfg) ≈ x / norm(x)
@test g_counter == 3

end

############
# Jacobian #
############

module Jacobian

using Base.Test
using ForwardDiff

global j_counter = 0
function f!(y, x)
    y[1] = x[1]*x[2]
    y[2] = x[2]*x[3]
    return y
end

function Df!(J, x)
    global j_counter += 1
    J[1,1] = x[2]
    J[1,2] = x[1]
    J[1,3] = 0
    J[2,1] = 0
    J[2,2] = x[3]
    J[2,3] = x[2]
    return J
end

J = zeros(2, 3)
x = rand(3)
y = zeros(2)
chunk = 2
cfig = ForwardDiff.JacobianImplementConfig(chunk, y, x)

@ForwardDiff.implement_jacobian! f! Df! cfig
cfg = ForwardDiff.JacobianConfig(nothing, y, x, ForwardDiff.Chunk{chunk}())
@test ForwardDiff.jacobian!(J, f!, y, x, cfg) == Df!(J, x)
@test j_counter == 3 # Two for chunk mode and one for the call after ==

# Test non mutating version
global j_counter2 = 0
f(x) = [x[1]*x[2], x[2]*x[3]]


function Df(x)
    global j_counter2 += 1
    return [x[2] x[1]   0;
            0    x[3] x[2]]
end

@ForwardDiff.implement_jacobian f Df
@test ForwardDiff.jacobian(f, x) == Df(x)
@test j_counter2 == 2 # One for jacobian and one for the call after ==
end

end
