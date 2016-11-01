module DeprecatedTest

using Base.Test
using ForwardDiff, DiffBase

include(joinpath(dirname(@__FILE__), "utils.jl"))

info("The following tests print lots of deprecation warnings on purpose.")

#############################################
# ForwardDiffResult --> DiffBase.DiffResult #
#############################################

v = rand()
x, y = rand(5), rand(5)
h = rand(5, 5)

@test isa(DerivativeResult(v, y), DiffBase.DiffResult)
@test isa(DerivativeResult(v), DiffBase.DiffResult)

@test isa(GradientResult(v, y), DiffBase.DiffResult)
@test isa(GradientResult(x), DiffBase.DiffResult)

@test isa(JacobianResult(x, y), DiffBase.DiffResult)
@test isa(JacobianResult(x), DiffBase.DiffResult)

@test isa(HessianResult(v, y, h), DiffBase.DiffResult)
@test isa(HessianResult(x), DiffBase.DiffResult)

######################
# gradient/gradient! #
######################

x = rand(5)
f = x -> prod(x) + sum(x)
v = f(x)
g = ForwardDiff.gradient(f, x)

@test ForwardDiff.gradient(f, x, Chunk{1}(); multithread = false) == g
@test ForwardDiff.gradient(f, x, Chunk{1}(); multithread = true) == g

out = similar(x)
ForwardDiff.gradient!(out, f, x, Chunk{1}(); multithread = false)
@test out == g

out = similar(x)
ForwardDiff.gradient!(out, f, x, Chunk{1}(); multithread = true)
@test out == g

out = DiffBase.GradientResult(x)
ForwardDiff.gradient!(out, f, x, Chunk{1}(); multithread = false)
@test DiffBase.value(out) == v
@test DiffBase.gradient(out) == g

out = DiffBase.GradientResult(x)
ForwardDiff.gradient!(out, f, x, Chunk{1}(); multithread = true)
@test DiffBase.value(out) == v
@test DiffBase.gradient(out) == g

######################
# jacobian/jacobian! #
######################

# f(x) -> y #
#-----------#

x = rand(5)
f = cumprod
y = f(x)
j = ForwardDiff.jacobian(f, x)

@test ForwardDiff.jacobian(f, x, Chunk{1}(); multithread = false) == j
@test ForwardDiff.jacobian(f, x, Chunk{1}(); multithread = true) == j

out = similar(x, length(y), length(x))
ForwardDiff.jacobian!(out, f, x, Chunk{1}(); multithread = false)
@test out == j

out = similar(x, length(y), length(x))
ForwardDiff.jacobian!(out, f, x, Chunk{1}(); multithread = true)
@test out == j

out = DiffBase.JacobianResult(x)
ForwardDiff.jacobian!(out, f, x, Chunk{1}(); multithread = false)
@test DiffBase.value(out) == y
@test DiffBase.jacobian(out) == j

out = DiffBase.JacobianResult(x)
ForwardDiff.jacobian!(out, f, x, Chunk{1}(); multithread = true)
@test DiffBase.value(out) == y
@test DiffBase.jacobian(out) == j

# f!(y, x) #
#----------#

y = similar(x)
f! = cumprod!
f!(y, x)
j = ForwardDiff.jacobian(f!, y, x)

@test ForwardDiff.jacobian(f!, y, x, Chunk{1}(); multithread = false) == j
@test ForwardDiff.jacobian(f!, y, x, Chunk{1}(); multithread = true) == j

out = similar(x, length(y), length(x))
ForwardDiff.jacobian!(out, f!, y, x, Chunk{1}(); multithread = false)
@test out == j

out = similar(x, length(y), length(x))
ForwardDiff.jacobian!(out, f!, y, x, Chunk{1}(); multithread = true)
@test out == j

out = DiffBase.JacobianResult(y, x)
ForwardDiff.jacobian!(out, f!, y, x, Chunk{1}(); multithread = false)
@test DiffBase.value(out) == y
@test DiffBase.jacobian(out) == j

out = DiffBase.JacobianResult(y, x)
ForwardDiff.jacobian!(out, f!, y, x, Chunk{1}(); multithread = true)
@test DiffBase.value(out) == y
@test DiffBase.jacobian(out) == j

####################
# hessian/hessian! #
####################

x = rand(5)
f = x -> prod(x) + sum(x)
v = f(x)
g = ForwardDiff.gradient(f, x)
h = ForwardDiff.hessian(f, x, Chunk{1}())

@test ForwardDiff.hessian(f, x, Chunk{1}(); multithread = false) == h
@test ForwardDiff.hessian(f, x, Chunk{1}(); multithread = true) == h

out = similar(x, length(x), length(x))
ForwardDiff.hessian!(out, f, x, Chunk{1}(); multithread = false)
@test out == h

out = similar(x, length(x), length(x))
ForwardDiff.hessian!(out, f, x, Chunk{1}(); multithread = true)
@test out == h

out = DiffBase.HessianResult(x)
ForwardDiff.hessian!(out, f, x, Chunk{1}(); multithread = false)
@test DiffBase.value(out) == v
@test DiffBase.gradient(out) == g
@test DiffBase.hessian(out) == h

out = DiffBase.HessianResult(x)
ForwardDiff.hessian!(out, f, x, Chunk{1}(); multithread = true)
@test DiffBase.value(out) == v
@test DiffBase.gradient(out) == g
@test DiffBase.hessian(out) == h

info("Deprecation testing is now complete, so any further deprecation warnings are real.")

end # module
