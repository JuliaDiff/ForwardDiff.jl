Limitations of ForwardDiff
==========================

ForwardDiff works by injecting user code with new number types that collect derivative
information at runtime. Naturally, this technique has some limitations. Here's a list
of all the roadblocks we've seen users run into ("target function" here refers to the
function being differentiated):

- **The target function can only be composed of generic Julia functions.** ForwardDiff cannot propagate derivative information through non-Julia code. Thus, your function may not work if it makes calls to external, non-Julia programs, e.g. uses explicit BLAS calls instead of ``Ax_mul_Bx``-style functions.

- **The target function must be unary (i.e., only accept a single argument).** There is an exception to this rule for ForwardDiff's ``jacobian`` API; see `the API documentation <api.html>`__ for details.

- **The target function must be written generically enough to accept numbers of type ``T<:Real`` as input  (or arrays of these numbers).** The function doesn't require a specific type signature, as long as the type signature is generic enough to avoid breaking this rule. This also means that any storage assigned used within the function must be generic as well (see `this comment`_ for an example).

- **Nested differentiation of closures is dangerous.** Differentiating closures is safe, and nested differentation is safe, but you might be vulnerable to a subtle bug if you try to do both. See `the relevant issue`_ for details.

- **The types of array inputs must be subtypes of** ``AbstractArray`` **.** Non-``AbstractArray`` array-like types are not  officially supported. However, these types might very well work if they overload ``ForwardDiff.replace_eltype{A,S}(x::A, ::Type{S})`` (whose definition can be found in `this file`_), which returns the type of ``x`` where the element type is replaced with ``S``.

.. _`this comment`: https://github.com/JuliaDiff/ForwardDiff.jl/issues/136#issuecomment-237941790
.. _`the relevant issue`: https://github.com/JuliaDiff/ForwardDiff.jl/issues/83
.. _`this file`: https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/src/cache.jl
