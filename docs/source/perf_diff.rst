Performing Differentiation
==========================

Restrictions on the target function
-----------------------------------

ForwardDiff.jl can only differentiate functions that adhere to the following rules:

- **The function must be unary (i.e., only accept a single argument).** The ``jacobian`` function is the exception to this restriction; see below for details.

- **The function must accept an argument whose type is a subtype of** ``Vector`` **or** ``Real``. The argument type **does not need to be annotated** in the function definition. In fact, **the function's argument type cannot be too restrictively annotated.** In this case, "too restrictive" means more restrictive than ``x::Vector`` or ``x::Number``.

- **All number types involved in the function must be subtypes of** ``Real``. We believe extension to subtypes of ``Complex`` is possible, but it hasn't yet been worked on.

- **The function must be** `type-stable`_ **.** This is not a strict limitation in every case, but in some cases, lack of type-stability can cause errors. At the very least, type-instablity can severely hinder performance.

.. _`type-stable`: http://julia.readthedocs.org/en/latest/manual/performance-tips/#write-type-stable-functions

Derivatives
-----------

ForwardDiff.jl can take derivatives of functions/callable objects of the form ``f(x::Number)`` → ``Number`` or ``f(x::Number)`` → ``Array``.

.. function:: derivative!(output::Array, f, x::Number)
    
    Compute :math:`f'(x)`, storing the output in ``output``.

.. function:: derivative(f, x::Number)

    Compute :math:`f'(x)`.

.. function:: derivative(f; mutates=false)
    
    Return the function :math:`f'`. If ``mutates=false``, then the returned function has the form ``d(x)``. If ``mutates=true``, then the returned function has the form ``d!(output, x)``.

Gradients
---------

.. note::

    When calling ForwardDiff.jl's ``gradient`` method, you must use the fully qualified method name ``ForwardDiff.gradient(...)`` in order to avoid conflict with ``Base.gradient``.

ForwardDiff.jl can take gradients of functions/callable objects of the form ``f(x::Vector)`` → ``Number``.

.. function:: gradient!(output::Vector, f, x::Vector)

    Compute :math:`\nabla f(\vec{x})`, storing the output in ``output``.

.. function:: ForwardDiff.gradient{S}(f, x::Vector{S})

    Compute :math:`\nabla f(\vec{x})`, where ``S`` is the element type of both the input and output.

.. function:: ForwardDiff.gradient(f; mutates=false)

    Return the function :math:`\nabla f`. If ``mutates=false``, then the returned function has the form ``g(x)``. If ``mutates=true``, then the returned function has the form ``g!(output, x)``.

Jacobians
---------

ForwardDiff.jl can take Jacobians of functions/callable objects of the form ``f(x:Vector)`` → ``Vector``.

.. function:: jacobian!(output::Matrix, f, x::Vector)

    Compute :math:`\mathbf{J}(f)(\vec{x})`, storing the output in ``output``.

.. function:: jacobian{S}(f, x::Vector{S})

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where ``S`` is the element type of both the input and output.

.. function:: jacobian(f; mutates=false, output_length=0)

    Return the function :math:`\mathbf{J}(f)`. If ``mutates=false``, then the returned function has the form ``j(x)``. If ``mutates=true``, then the returned function has the form ``j!(output, x)``.

    This method is special in that it supports target functions of the form ``f!{T}(output::Vector{T}, x::Vector{T})``, where ``output`` stores the result. To utilize this functionality, pass the target function in as usual, and set ``output_length`` to the expected length of ``output``.

Hessians
--------

ForwardDiff.jl can take Hessians of functions/callable objects of the form ``f(x::Vector)`` → ``Number``.

.. function:: hessian!(output::Matrix, f, x::Vector)

    Compute :math:`\mathbf{H}(f)(\vec{x})`, storing the output in ``output``.

.. function:: hessian{S}(f, x::Vector{S})

    Compute :math:`\mathbf{H}(f)(\vec{x})`, where ``S`` is the element type of both the input and output.

.. function:: hessian(f; mutates=false)

    Return the function :math:`\mathbf{H}(f)`. If ``mutates=false``, then the returned function has the form ``h(x)``. If ``mutates=true``, then the returned function has the form ``h!(output, x)``.

Tensors
-------

ForwardDiff.jl can take tensors of functions/callable objects of the form ``f(x::Vector)`` → ``Number``.

The word "tensor", in this context, refers to a :math:`3^{\text{rd}}` order generalization of the Hessian. Given a function :math:`f:\mathbb{R}^n \to \mathbb{R}`, the tensor operator :math:`\mathbf{T}` is defined as

.. math::
    
    \mathbf{T}(f) = \sum_{i,j,k=1}^{n} \frac{\delta^3 f}{\delta x_i \delta x_j \delta x_k}

.. function:: tensor!(output::Matrix, f, x::Vector)

    Compute :math:`\mathbf{T}(f)(\vec{x})`, storing the output in ``output``.

.. function:: tensor{S}(f, x::Vector{S})

    Compute :math:`\mathbf{T}(f)(\vec{x})`, where ``S`` is the element type of both the input and output.

.. function:: tensor(f; mutates=false)

    Return the function :math:`\mathbf{T}(f)`. If ``mutates=false``, then the returned function has the form ``t(x)``. If ``mutates=true``, then the returned function has the form ``t!(output, x)``.
