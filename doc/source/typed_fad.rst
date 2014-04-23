Type-Based FAD
================================================================================

Type-based FAD is invoked by setting the keyword argument *fadtype* to *:typed* in the relevant API functions of the
package. The function *f* to be differentiated has the same signature *f(x::Vector)* across all type-based FAD methods.

First-Order Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Derivatives and Gradients
---------------------------------------------------------------------------------

The following two API functions are used for computing the gradient of a function
:math:`f:\mathbb{R}^n\rightarrow\mathbb{R}` or its derivative in the case of :math:`n=1`:

- *forwarddiff_gradient(f, ::Type, fadtype=:typed)*,
- *forwarddiff_gradient!(f, ::Type, fadtype=:typed)*.

*::Type* specifies the type of involved arithmetic. Typically, this argument is set to *Float64*.

*forwarddiff_gradient* returns the gradient in the form of a function *g(x::Vector)*. On the other hand,
*forwarddiff_gradient!* returns the gradient as a function with signature *g!(x::Vector, y::Vector)*, where *x* and *y*
hold the input and output vectors respectively.

To exemplify usage, consider computhing the derivative :math:`g(x)=x` of :math:`f(x)=x^2`.

.. code-block:: julia
  :linenos:

  using ForwardDiff

  f(x) = x[1]^2

  # Using forwarddiff_gradient
  g = forwarddiff_gradient(f, Float64, fadtype=:typed)
  g([2.])

  # Using forwarddiff_gradient!
  g! = forwarddiff_gradient!(f, Float64, fadtype=:typed)
  y = Array(Float64, 1)
  g!([2.], y)

Jacobians
---------------------------------------------------------------------------------

To compute the Jacobian of a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m` using typed-based FAD, the two
following methods can be used:

- *forwarddiff_jacobian(f, ::Type, fadtype=:typed)*,
- *forwarddiff_jacobian!(f, ::Type, fadtype=:typed)*.

Note that these functions are aliases of the corresponding *forwarddiff_gradient* and *forwarddiff_gradient*, therefore
their invocation is not altered. The Jacobian functions returned by *forwarddiff_jacobian* and *forwarddiff_jacobian!*
have the respective forms *j(x::Vector)* and *j!(x::Vector, y::Matrix)*, where *y* holds the evaluated Jacobian.

Second-Order Derivatives (Hessians)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Hessian matrix of a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}` is computed via typed-based FAD by the
methods

- *forwarddiff_hessian(f, ::Type, fadtype=:typed)*,
- *forwarddiff_hessian!(f, ::Type, fadtype=:typed)*.

*forwarddiff_hessian* returns the Hessian *h(x::Vector)*, while the signature of the Hessian function returned by
*forwarddiff_hessian!* is *h!(x::Vector, y::Matrix)*, *x* and *y* being the input vector and output matrix respectively.

For example, consider the function :math:`f:\mathbb{R}^2\rightarrow\mathbb{R}` defined as
:math:`f(x, y) = x^2y^2+x^3`. Its Hessian is given by

.. math::
  :label: `hessianex01`

  H_f(x, y) =
  \left(\begin{matrix} 6x+2y^2 & 4xy \\
    4xy & 2x^2
  \end{matrix}\right)

The code for computing :math:`J_f(2.1,1.5)` using typed-based FAD is provided below:

.. code-block:: julia
  :linenos:

  using ForwardDiff

  f(x) = x[1]^2*x[2]^2+x[1]^3

  # Using forwarddiff_hessian
  h = forwarddiff_hessian(f, Float64, fadtype=:typed)

  h([1.5, -3.1])
  # 2x2 Array{Float64,2}:
  #   28.22  -18.6
  #  -18.6     4.5

  # Using forwarddiff_hessian!
  h! = forwarddiff_hessian!(f, Float64, fadtype=:typed)

  y = zeros(2, 2)
  h!([1.5, -3.1], y)
  # 2x2 Array{Float64,2}:
  #   28.22  -18.6
  #  -18.6     4.5

Third-Order Derivatives (Tensors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The partial derivatives of a Hessian matrix, commonly referred to as tensors, of a function
:math:`f:\mathbb{R}^n\rightarrow\mathbb{R}` are computed via typed-based FAD by the following API:

- *forwarddiff_tensor(f, ::Type, fadtype=:typed)*,
- *forwarddiff_tensor!(f, ::Type, fadtype=:typed)*.

*forwarddiff_tensor* returns the tensor function in the form *t(x::Vector)*, while *forwarddiff_tensor!* returns the
tensor function with signature *t(x::Vector, y::Array)*, where *x* is the *n*-length input vector and *y* is the
:math:`n\times n \times n` output array.

For instance, the tensors of :math:`f(x, y) = x^2y^2+x^3` are

.. math::
  :label: `tensorex01`

  \frac{\partial H_f(x, y)}{\partial x} =
  \left(\begin{matrix} 6 & 4y \\
    4y & 4x
  \end{matrix}\right),~
  \frac{\partial H_f(x, y)}{\partial y} =
  \left(\begin{matrix} 4y & 4x \\
    4x & 0
  \end{matrix}\right).

To compute :math:`\frac{\partial H_f(x, y)}{\partial x}` and :math:`\frac{\partial H_f(x, y)}{\partial y}` via
typed-based FAD, the API is used as follows:

.. code-block:: julia
  :linenos:

  using ForwardDiff

  f(x) = x[1]^2*x[2]^2+x[1]^3

  # Using forwarddiff_tensor
  t = forwarddiff_tensor(f, Float64, fadtype=:typed)

  t([1.5, -3.1])
  # 2x2x2 Array{Float64,3}:
  # [:, :, 1] =
  #    6.0  -12.4
  #  -12.4    6.0
  #
  # [:, :, 2] =
  #  -12.4  6.0
  #    6.0  0.0

  # Using forwarddiff_tensor!
  t! = forwarddiff_tensor!(f, Float64, fadtype=:typed)

  y = zeros(2, 2, 2)
  t!([1.5, -3.1], y)
  y
  # 2x2x2 Array{Float64,3}:
  # [:, :, 1] =
  #    6.0  -12.4
  #  -12.4    6.0
  #
  # [:, :, 2] =
  #  -12.4  6.0
  #    6.0  0.0
