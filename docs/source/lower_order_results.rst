Accessing Lower-Order Results
=============================

The Wrong Way
-------------

Let's say you want to calculate the value, gradient, and Hessian of some function ``f`` at an input ``x``.

You might simply do the following:

.. code-block:: julia

    julia> import ForwardDiff

    # a silly example function
    julia> f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);

    julia> x = rand(4); # a silly input vector

    julia> val = f(x)
    6.8401101379076685

    julia> grad = ForwardDiff.gradient(f, x)
    4-element Array{Float64,1}:
     10.7476
      9.32271
      9.47858
      9.50792

    julia> hess = ForwardDiff.hessian(f, x)
    4x4 Array{Float64,2}:
     13.082   21.2597  21.8144  21.9039
     21.2597  18.9436  19.0885  19.1668
     21.8144  19.0885  22.928   19.6656
     21.9039  19.1668  19.6656  23.4095

**The above is actually a horribly redundant way to accomplish this task!** This is because, in the course of calculating higher-order derivatives, **ForwardDiff.jl ends up calculating all the lower-order derivatives anyway.**

The Right Way
-------------

You can pass the abstract type ``AllResults`` to a differentiation method in order to return a "package" of all the lower-order calculations along with the normal result of the method.

Let's use the situation from the previous section as an example. Here's the *right* way to get the Hessian and all the lower-order values:

.. code-block:: julia

    julia> hess, allresults = ForwardDiff.hessian(f, x, ForwardDiff.AllResults);

    julia> hess
    4x4 Array{Float64,2}:
     13.082   21.2597  21.8144  21.9039
     21.2597  18.9436  19.0885  19.1668
     21.8144  19.0885  22.928   19.6656
     21.9039  19.1668  19.6656  23.4095

    julia> val = ForwarDiff.value(allresults)
    6.8401101379076685

    julia> grad = ForwardDiff.gradient(allresults)
    4-element Array{Float64,1}:
     10.7476
      9.32271
      9.47858
      9.50792

As you can see, passing ``AllResults`` to ``hessian`` causes it to return two objects instead of one. The first is simply the Hessian that you'd normally expect, but the second is quite different. It's an instance of the ``ForwardDiffResult`` type, which packages all of the calculation's lower-order results for easy access. As the above example demonstrates, lower-order results are obtained by calling the relevant differentation method on the ``ForwardDiffResult`` instance.

Mutating methods can also be used, if you want to load a result into an existing array:

.. code-block:: julia

    julia> output = Array(Float64, 4);

    # fill output with the gradient extracted from allresults
    julia> ForwardDiff.gradient!(output, allresults);

    julia> output
    4-element Array{Float64,1}:
     10.7476
      9.32271
      9.47858
      9.50792

What You Can Extract from a ``ForwardDiffResult``
-------------------------------------------------

The below table describes the possible results that can be extracted given the differentiation method used:

+-------------------------------------------------+---------------------------------------------------------------------+
| You called the method...                        | Available extraction methods for the returned ``ForwardDiffResult`` |
+=================================================+=====================================================================+
| derivative(f, x, AllResults)                    | value(::ForwardDiffResult)                                          |
|                                                 |                                                                     |
| derivative!(output, f, x, AllResults)           | value!(::Array, ::ForwardDiffResult)                                |
|                                                 |                                                                     |
|                                                 | derivative(::ForwardDiffResult)                                     |
|                                                 |                                                                     |
|                                                 | derivative!(::Array, ::ForwardDiffResult)                           |
+-------------------------------------------------+---------------------------------------------------------------------+
| ForwardDiff.gradient(f, x, AllResults)          | value(::ForwardDiffResult)                                          |
|                                                 |                                                                     |
| ForwardDiff.gradient!(output, f, x, AllResults) | value!(::Array, ::ForwardDiffResult)                                |
|                                                 |                                                                     |
|                                                 | ForwardDiff.gradient(::ForwardDiffResult)                           |
|                                                 |                                                                     |
|                                                 | ForwardDiff.gradient!(::Array, ::ForwardDiffResult)                 |
+-------------------------------------------------+---------------------------------------------------------------------+
| jacobian(f, x, AllResults)                      | value(::ForwardDiffResult)                                          |
|                                                 |                                                                     |
| jacobian!(output, f, x, AllResults)             | value!(::Array, ::ForwardDiffResult)                                |
|                                                 |                                                                     |
|                                                 | jacobian(::ForwardDiffResult)                                       |
|                                                 |                                                                     |
|                                                 | jacobian!(::Array, ::ForwardDiffResult)                             |
+-------------------------------------------------+---------------------------------------------------------------------+
| hessian(f, x, AllResults)                       | value(::ForwardDiffResult)                                          |
|                                                 |                                                                     |
| hessian!(output, f, x, AllResults)              | value!(::Array, ::ForwardDiffResult)                                |
|                                                 |                                                                     |
|                                                 | gradient(::ForwardDiffResult)                                       |
|                                                 |                                                                     |
|                                                 | gradient!(::Array, ::ForwardDiffResult)                             |
|                                                 |                                                                     |
|                                                 | hessian(::ForwardDiffResult)                                        |
|                                                 |                                                                     |
|                                                 | hessian!(::Array, ::ForwardDiffResult)                              |
+-------------------------------------------------+---------------------------------------------------------------------+
| tensor(f, x, AllResults)                        | value(::ForwardDiffResult)                                          |
|                                                 |                                                                     |
| tensor!(output, f, x, AllResults)               | value!(::Array, ::ForwardDiffResult)                                |
|                                                 |                                                                     |
|                                                 | gradient(::ForwardDiffResult)                                       |
|                                                 |                                                                     |
|                                                 | gradient!(::Array, ::ForwardDiffResult)                             |
|                                                 |                                                                     |
|                                                 | hessian(::ForwardDiffResult)                                        |
|                                                 |                                                                     |
|                                                 | hessian!(::Array, ::ForwardDiffResult)                              |
|                                                 |                                                                     |
|                                                 | tensor(::ForwardDiffResult)                                         |
|                                                 |                                                                     |
|                                                 | tensor!(::Array, ::ForwardDiffResult)                               |
+-------------------------------------------------+---------------------------------------------------------------------+
