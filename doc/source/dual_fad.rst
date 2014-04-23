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
type-based FAD. As a reminder, the *f* argument of *forwarddiff_gradient* has signature *f(x::Vector)*, whereas the
signature of the *f* argument of *forwarddiff_gradient!* is *f(x::Vector, y::Vector)*. The mere change observed in the
dual API in comparison to type-based FAD is the extra keyword argument *n*, which specifies the dimension of the
function's domain.

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

Functions of Mutlivariate Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Jacobian of a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m,` is computed via fual FAD by invoking either of
the following functions:

- forwarddiff_gradient(f!, ::Type, fadtype=:dual, n=1, m=1),
- forwarddiff_gradient!(f!, ::Type, fadtype=:dual, n=1, m=1).

