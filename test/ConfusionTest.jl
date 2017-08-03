module ConfusionTest

using Base.Test
using ForwardDiff

using Base.Test


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


end # module
