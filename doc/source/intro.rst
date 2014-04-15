Introduction
================================================================================

Forward-mode automatic differentiation (FAD) can be implemented in various ways. The present package aims at realizing
several FAD implementations in Julia. The main motivation behind this aim is to utilize the heterogeneous advantages of
the different coding approaches to FAD.

FAD Implementations
---------------------------------------------------------------------------------

A synopsis of the FAD implementations of the package is set out below.

Type-Based FAD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One FAD implementation of the package defines the types *GraDual*, *FADHessian* and *FADTensor* to compute the 
respective first, second and third-order derivatives of functions :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m`. More
specifically, *GraDual* is used for deriving via FAD gradients and Jacobians, *FADHessian* for Hessians and *FADTensor*
for tensors, that is derivatives of Hessians. Despite its relatively slow speed, type-based FAD finds its utility in the
computation of up to third-order derivatives and, being a well-tested FAD suite itself, in the cross-testing of other
FAD methods.

The *GraDual*, *FADHessian* and *FADTensor* types are used internally by the package. The user is not required to
instantiate them, since the interface operates at a higher level requiring to define the function to be differentiated.

FAD Using Dual Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another coded FAD approach makes use of dual numbers, which are represented by the *Dual* type in the *DualNumbers*
package. This approach is suitable for computing only first-order derivatives of functions
:math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m` and does not have the capacity of the type-based approach to compute
higher-order derivatives. On the other hand, the dual-based approach is thought to be faster than the type-based one 
for FAD of gradients and Jacobians. Benchmarks will be provided once all four FAD methods are implemented.

The user does not have to interfere with `Dual` numbers given that the higher-level of the API mainly requires the
differentiable function to be inputted.

Matrix FAD Using Kronecker and Box Products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To be implemented.

Power Series FAD Using Generalized Dual Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To be implemented.

The Main API
---------------------------------------------------------------------------------

The API consists of functions that in principle take a differentiable function as input and return the k-th order
derivative of the function. An abstract description of the API is provided, followed by examples illustrating the
concepts and usage.

API Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three components synthesize the API function names:

- All API routines start with the common prefix *forwarddiff\_*.
- The second part determines the order k of the outputted derivative function. In particular, valid choices for the
  second component include *gradient*, *jacobian*, *hessian* and *tensor*.
- The third and final part of an API function name is an exclamation mark *!*, which may or may not be present. This
  does not indicate if any API input is modified, but instead informs whether the returned derivative function modifies
  its input.

Table 1 displays all the API functions, demonstrating which of the package FAD approaches provide the relevant
functionality.

+-----------------------+-------------------------+ 
| API Function          | FAD Method              | 
|                       +-------------+-----------+
|                       | Type-Based  | Dual      | 
+-----------------------+-------------+-----------+ 
| forwarddiff_gradient! | Yes         | Yes       | 
+-----------------------+-------------+-----------+ 
| forwarddiff_gradient  | Yes         | Yes       | 
+-----------------------+-------------+-----------+ 
| forwarddiff_jacobian! | Yes         | Yes       | 
+-----------------------+-------------+-----------+ 
| forwarddiff_jacobian  | Yes         | Yes       | 
+-----------------------+-------------+-----------+ 
| forwarddiff_hessian!  | Yes         | No        | 
+-----------------------+-------------+-----------+ 
| forwarddiff_hessian   | Yes         | No        | 
+-----------------------+-------------+-----------+ 
| forwarddiff_tensor!   | Yes         | No        | 
+-----------------------+-------------+-----------+ 
| forwarddiff_tensor    | Yes         | No        | 
+-----------------------+-------------+-----------+ 

**Table 1**: API functions and their availability via the package's various FAD implementations.

Input Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every API function takes two required arguments:

- The first required argument is the function to be differentiated and is therefore of type *Function*. The anticipated
  format of the input function varies depending on the FAD approach. In general the input function can take two possible
  forms; it can be either *f(x)* with one argument representing the function input or *f!(x, y)*, in which case *x*
  holds the function input and y is a dummy variable meant to receive the function output. For instance, for a function
  :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m` the input *x* of *f(x)* is expected to be a vector.
- The second required argument holds the type of the differentiable function's domain and range, and consequently the
  type of the returned derivatives. For example, it can take values such such *Real* or *Float64*.

The keyword argument *fadtype* succeeds the two required ones. *fadtype* can take one of the values *:typed*, *:dual*,
*:box* or *:pseries* to indicate the preferred FAD method. It is noted that each API function restricts the available
values for *fadtype* as explained in Table 1. Furthermore, the *:box* and *:pseries* values will be released once
the associated FAD methods are implemented.

Any options specific to a FAD method are passed to the API as additional keyword arguments.

Output Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every API function returns a derivative as a function. The name of the API method prerscribes the format of the
returned derivative. To elaborate, presume a function :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}` with gradient
:math:`g(x):=\nabla f(x)`. Then *forwarddif_gradient* returns the function *g(x::Vector)*, whereas
*forwarddif_gradient!* returns *g!(x::Vector, y::Vector)*, where *y* is designated to store the FAD value of
the gradient. In other words, *g(x)* outputs the gradient value via a *return* statement while *g!(x, y)* saves the
gradient in *y*.

Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a first illustration of the interface, consider the function :math:`f:\mathbb{R}^2\rightarrow\mathbb{R}^3` defined
by :math:`f(x, y) = (x^2+y, 3x, x^2y^3)`. Interest is in computing the Jacobian of *f*, which is given
by

.. math::
  :label: jac_ex1

  J_f(x, y) =
  \left(\begin{matrix} 2x & 1 \\
    3 & 0 \\
    2xy^3 & 3(xy)^2 \end{matrix}\right)

The code for computing :math:`J_f(2.1,1.5)` using typed-based FAD is provided below:

.. code-block:: julia

  using ForwardDiff

  f(x) = [x[1]^2+x[2], 3*x[1], x[1]^2*x[2]^3]
  g = forwarddiff_jacobian(f, Float64, fadtype=:typed)

  g([2.1, 1.5])
