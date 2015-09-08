How ForwardDiff.jl Takes Derivatives
====================================

As previously stated, ForwardDiff.jl is an implementation of `forward mode automatic differentiation`_ (AD) in Julia. There are two key components of this implementation: number types, and the API.

.. _`forward mode automatic differentiation`: https://en.wikipedia.org/wiki/Automatic_differentiation

``ForwardDiffNumber`` s
-----------------------

ForwardDiff.jl provides several new number types, which are all subtypes of ``ForwardDiffNumber{N,T,C} <: Number``. These number types store both normal values, and the values of partial derivatives.

Elementary numerical functions on these types are overloaded to evaluate both the original function, *and* evaluate partials derivatives of the function. Julia's multiple dispatch then allows us to pass these number types into a general function :math:`f` (which is assumed to be composed of the overloaded elementary functions), and the derivative information is naturally propogated at each step of the calculation by way of the chain rule.

This propogation occurs all the way through to the result of the function, which finally contains :math:`f(x)` and :math:`f'(x)`, where :math:`x` was the original point of evalutation.

ForwardDiff.jl's API
--------------------

The second component provided by this package is the API, which abstracts away the number types and makes it easy to obtain the results of familiar calculations like taking the Jacobian or Hessian. This way, users don't have to understand ``ForwardDiffNumber`` s in order to make use of the package.

The API also provides features like configurable ``chunk_size``s for fine-tuning calculations without having to hack in new implementations of the API methods.
