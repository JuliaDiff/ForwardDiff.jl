Introduction
================================================================================

Forward-mode automatic differentiation (FAD) can be implemented in various ways. The present package aims at realizing
several FAD implementations in Julia. The main motivation behind this aim is to utilize the heterogeneous advantages of
the different coding approaches to FAD.

FAD Implementations
---------------------------------------------------------------------------------

A synopsis of the FAD implementations of the package is set out below.

**Type-Based FAD**

One FAD implementation of the package defines the types *GraDual*, *FADHessian* and *FADTensor* to compute the 
respective first, second and third-order derivatives of functions :math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m`. More
specifically, *GraDual* is used for deriving via FAD gradients and Jacobians, *FADHessian* for Hessians and *FADTensor*
for tensors, that is derivatives of Hessians. Despite its relatively slow speed, type-based FAD finds its utility in the
computation of up to third-order derivatives and, being a well-tested FAD suite itself, in the cross-testing of other
FAD methods.

The *GraDual*, *FADHessian* and *FADTensor* types are used internally by the package. The user is not required to
instantiate them, since the interface operates at a higher level requiring to define the function to be differentiated.

**FAD Using Dual Numbers**

Another available FAD approach makes use of dual numbers, which are represented by the *Dual* type in the *DualNumbers*
package. This approach is suitable for computing only first-order derivatives of functions
:math:`f:\mathbb{R}^n\rightarrow\mathbb{R}^m` and does not have the capacity of the type-based approach to compute
higher-order derivatives. On the other hand, the dual-based approach is thought to be faster than the type-based one 
for FAD of gradients and Jacobians. Benchmarks will be provided once all four FAD methods are implemented.

The user does not have to interfere with `Dual` numbers given that the higher-level of the API mainly requires the
differentiable function to be inputted.

**Matrix FAD Using Kronecker and Box Products**

To be implemented.

**Power Series FAD Using Generalized Dual Numbers**

To be implemented.

The Main API
---------------------------------------------------------------------------------

The API consists of functions that in principle take a differentiable function as input and return the k-th order
derivative of the function.

Input Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To appear.

Output Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To appear.

API Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three components synthesize the API function names:

- All API routines start with the common prefix *forwarddiff\_*.
- The second part determines the order k of the outputted derivative function. In particular, valid choices for the
  second component include *gradient*, *jacobian*, *hessian* and *tensor*.
- The third and final part of an API function name is an exclamation mark *!*, which may or may not be present. This
  does not indicate if any API input is modified, but instead informs whether the returned derivative function modifies
  its input.

Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To appear.
