Overview: How ForwardDiff.jl Works
==================================

As previously stated, ForwardDiff.jl is an implementation of `forward mode automatic differentiation`_ (AD) in Julia. There are two key components of this implementation: number types, and the API.

.. _`forward mode automatic differentiation`: https://en.wikipedia.org/wiki/Automatic_differentiation

New Number Types
----------------

ForwardDiff.jl provides several new number types, which are all subtypes of the abstract type ``ForwardDiffNumber{N,T,C} <: Number``. These number types store both normal values, and the values of partial derivatives.

Elementary numerical functions on these types are overloaded to evaluate both the original function, *and* evaluate partials derivatives of the function, storing the results in a ``ForwardDiffNumber``. We can then pass these number types into a general function :math:`f` (which is assumed to be composed of the overloaded elementary functions), and the derivative information is naturally propogated at each step of the calculation by way of the chain rule.

This propogation occurs all the way through to the result of the function, which is iself a ``ForwardDiffNumber`` or an ``Array{ForwardDiffNumber}``. This number (or array) contains the result :math:`f(x)` and the derivative :math:`f'(x)`, where :math:`x` was the original point of evalutation.

ForwardDiff.jl's API
--------------------

The second component provided by this package is the API, which abstracts away the number types and makes it easy to execute familiar calculations (e.g. taking the Jacobian, Hessian, etc.). This way, users don't have to understand ``ForwardDiffNumber`` types in order to make use of the package.

The API also provides features (like the ``chunk_size`` option) for fine-tuning calculations without having to hack in new implementations of the API methods.
