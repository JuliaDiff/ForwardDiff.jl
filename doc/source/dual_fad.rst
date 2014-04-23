FAD Using Dual Numbers
================================================================================

Dual-based FAD is invoked by setting the keyword argument *fadtype* to *:dual* in the API function calls related to
first-order derivatives, i.e. gradients and Jacobians. Due to the heterogeneous underlying implementation, the API
for dual FAD differs between functions of univariate and multivariate range.

Functions of Univariate Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To derive the dual FAD gradient of a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}`, one can employ either of the
two functions

- forwarddiff_gradient(f, ::Type, fadtype=:dual, n=1),
- forwarddiff_gradient!(f, ::Type, fadtype=:dual, n=1).

The anticipated format of the function *f* under differentiation is the same as the one encountered in the case of
type-based FAD. As a reminder, the signature of f is *f(x::Vector)*. The mere change observed in the dual API in
comparison to type-based FAD is the extra keyword argument *n*, which specifies the dimension of the function's domain.


The gradient function returned by *forwarddiff_gradient* has the form *g(x::Vector)*, whereas the gradient function
returned by *forwarddiff_gradient!* is of the form *g!(x::Vector, y::Vector)*, where the vector *y* holds the output,
that is the evaluated gradient. 

To compare the APIs of type-based and dual FAD, the corresponding example of the previous section is replicated, this
time by employing dual FAD to compute the derivative of :math:`f(x)=x^2`:

.. code-block:: julia
  :linenos:

  using ForwardDiff

  f(x) = x[1]^2

  # Using forwarddiff_gradient
  g = forwarddiff_gradient(f, Float64, fadtype=:dual)
  g([2.])

  # Using forwarddiff_gradient!
  g! = forwarddiff_gradient!(f, Float64, fadtype=:dual)
  y = Array(Float64, 1)
  g!([2.], y)

Functions of Multivariate Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Jacobian of a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m,` is computed via fual FAD by invoking either of
the following functions:

- forwarddiff_jacobian(f!, ::Type, fadtype=:dual, n=1, m=1),
- forwarddiff_jacobian!(f!, ::Type, fadtype=:dual, n=1, m=1).

One major difference in relation to the input specification of previous API methods, is that the function under
differentiation is not defined as *f(x::Vector)*, but it instead conforms to the signature *f!(x::Vector, y::Vector)*,
where :math:`x\in\mathbb{R}^n` and :math:`y\in\mathbb{R}^m` are the input and output vectors of *f!*. A second
difference is the additional keyword argument *m*, which indicates the dimension of the function's range.

The Jacobian function is returned in the same format as the one of the corresponding type-based FAD functions. So,
the Jacobians returned by *forwarddiff_jacobian* and by by *forwarddiff_gradient!* have the respective forms
*j(x::Vector)* and *j!(x::Vector, y::Matrix)*.

The example below demonstrates how to perform dual-based FAD for the calculation of the Jacobian. It is a replication of
the example in the introduction, where the Jacobian of :math:`f(x, y) = (x^2+y, 3x, x^2y^3)` was evaluated using
type-based FAD.

.. code-block:: julia
  :linenos:

  using ForwardDiff

  function f!(x, y)
    y[1] = x[1]^2+x[2]
    y[2] = 3*x[1]
    y[3] = x[1]^2*x[2]^3
  end

  # Using forwarddiff_jacobian
  j = forwarddiff_jacobian(f!, Float64, fadtype=:dual, n=2, m=3)

  j([2.1, 1.5])
  # 3x2 Array{Float64,2}:
  #   4.2     1.0   
  #   3.0     0.0   
  #  14.175  29.7675

  # Using forwarddiff_jacobian
  j! = forwarddiff_jacobian!(f!, Float64, fadtype=:dual, n=2, m=3)

  y = zeros(3, 2)
  j!([2.1, 1.5], y)
