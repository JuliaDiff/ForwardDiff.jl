# Limitations of ForwardDiff

ForwardDiff works by injecting user code with new number types that collect derivative
information at runtime. Naturally, this technique has some limitations. Here's a list
of all the roadblocks we've seen users run into ("target function" here refers to the
function being differentiated):

- **The target function can only be composed of generic Julia functions.** ForwardDiff cannot propagate derivative information through non-Julia code. Thus, your function may not work if it makes calls to external, non-Julia programs, e.g. uses explicit BLAS calls instead of `Ax_mul_Bx`-style functions.

- **The target function must be a composition of differentiable functions.** ForwardDiff can have issues to compute derivatives of functions, which are composed of at least one function, which is not differentiable in the point the derivative should be
evaluated, even if the target function itself is differentiable. A simple example is `f(x) = norm(x)^2`, where `ForwardDiff.gradient(f, zeros(2))` returns a vector of `NaN`s since the Euclidean norm is not differentiable in zero. A possible solution to
this issue is to, e.g., define `f(x) = sum(abs2, x)` instead. In situations, where rewriting the target function only as a composition of differentiable functions is more complicated (e.g. `f(x) = (1 + norm(x))exp(-norm(x))`)), one would need to define
a custom derivative rule (see [this comment](https://github.com/JuliaDiff/ForwardDiff.jl/issues/303#issuecomment-2977990425)).

- **The target function must be unary (i.e., only accept a single argument).** [`ForwardDiff.jacobian`](@ref) is an exception to this rule.

- **The target function must be written generically enough to accept numbers of type `T<:Real` as input (or arrays of these numbers).** The function doesn't require a specific type signature, as long as the type signature is generic enough to avoid breaking this rule. This also means that any storage assigned used within the function must be generic as well (see [this comment](https://github.com/JuliaDiff/ForwardDiff.jl/issues/136#issuecomment-237941790) for an example).

- **The types of array inputs must be subtypes of** `AbstractArray` **.** Non-`AbstractArray` array-like types are not officially supported.

ForwardDiff is not natively compatible with rules defined by the [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) ecosystem.
You can use [ForwardDiffChainRules.jl](https://github.com/ThummeTo/ForwardDiffChainRules.jl) to bridge this gap.
