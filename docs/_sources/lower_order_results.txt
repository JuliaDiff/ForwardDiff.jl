Accessing Lower-Order Results
=============================

The Wrong Way
-------------

Let's say you want to calculate the value, gradient, and Hessian of some function ``f`` at
an input ``x``. Here's the **wrong way** to do it:

.. code-block:: julia

    julia> using ForwardDiff

    # a silly example function
    julia> f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);

    # a silly input vector
    julia> x = rand(4);

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

**The above is a horribly redundant way to accomplish this task!** In the course of
calculating higher-order derivatives, ForwardDiff ends up calculating all the lower-order
derivatives and original value ``f(x)``. In the following section, we'll see how we can
use ``ForwardDiffResult`` types to efficiently access these results.

The Right Way
-------------

To retrieve all the lower-order calculations along with the normal result of an API
function, pass an instance of the appropriate ``ForwardDiffResult`` type to the in-place
version of the function. Let's use the situation from the previous section as an example.
Here's the *right* way to get the Hessian and all the lower-order values:

.. code-block:: julia

    julia> using ForwardDiff

    julia> f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);

    julia> x = rand(4);

    julia> result = HessianResult(x)

    julia> ForwardDiff.hessian!(result, f, x);

    julia> ForwardDiff.value(result) == f(x)
    true

    julia> ForwardDiff.gradient(result) == ForwardDiff.gradient(f, x)
    true

    julia> ForwardDiff.hessian(result) == ForwardDiff.hessian(f, x)
    true

The following accessor methods are available to extract data from ``ForwardDiffResult`` types:

.. code-block:: julia

    ForwardDiff.value(result::ForwardDiffResult)
    ForwardDiff.derivative(result::DerivativeResult)
    ForwardDiff.gradient(result::Union{GradientResult,HessianResult})
    ForwardDiff.jacobian(result::JacobianResult)
    ForwardDiff.hessian(result::HessianResult)

Here are the various constructors for all the ``ForwardDiffResult`` types (using ``new`` to
denote the default constructor):

.. code-block:: julia

    DerivativeResult(value_output, derivative_output) = new(value_output, derivative_output)
    DerivativeResult(x) = DerivativeResult(copy(x), copy(x))

    GradientResult(value_output, gradient_output) = new(value_output, gradient_output)
    GradientResult(x) = GradientResult(first(x), similar(x))

    JacobianResult(value_output, jacobian_output) = new(value_output, jacobian_output)
    JacobianResult(x) = JacobianResult(similar(x), similar(x, length(x), length(x)))

    HessianResult(value_output, gradient_output, hessian_output) = new(value_output, gradient_output, hessian_output)
    HessianResult(x) = HessianResult(first(x), similar(x), similar(x, length(x), length(x)))
