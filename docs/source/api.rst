ForwardDiff API
===============

Restrictions on the target function
-----------------------------------

ForwardDiff can only differentiate functions that adhere to the following rules:

- **The function can only be composed of generic Julia functions.** ForwardDiff cannot propagate derivative information through non-Julia code. Thus, your function may not work if it makes calls to external, non-Julia programs, e.g. uses explicit BLAS calls instead of ``Ax_mul_Bx``-style functions.

- **The function must be unary (i.e., only accept a single argument).** The ``jacobian`` function is the exception to this restriction; see below for details.

- **The function must accept an argument whose type is a subtype of** ``Vector`` **or** ``Real``. The argument type does not need to be annotated in the function definition.

- **The function's argument type cannot be too restrictively annotated.** In this case, "too restrictive" means more restrictive than ``x::Vector`` or ``x::Real``.

- **The function should be** `type-stable`_ **.** This is not a strict limitation in every case, but in some cases, lack of type-stability can cause errors. At the very least, type-instablity can severely hinder performance.

.. _`type-stable`: http://julia.readthedocs.org/en/latest/manual/performance-tips/#write-type-stable-functions

Derivatives of :math:`f(x) : \mathbb{R} \to \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k}`
--------------------------------------------------------------------------------------------------

.. function:: ForwardDiff.derivative!(out, f, x)

    Compute :math:`f'(x)`, storing the output in ``out``.

.. function:: ForwardDiff.derivative!(out::DerivativeResult, f, x)

    Compute :math:`f'(x)`, storing the output in ``out``. The value of :math:`f(x)` can be
    accessed by calling ``ForwardDiff.value(out)``, while :math:`f'(x)` can be accessed by
    calling ``ForwardDiff.derivative(out)``.

.. function:: ForwardDiff.derivative(f, x)

    Compute and return :math:`f'(x)`.

Gradients of :math:`f(x) : \mathbb{R}^n \to \mathbb{R}`
-------------------------------------------------------

.. function:: ForwardDiff.gradient!(out, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false)

    Compute :math:`\nabla f(\vec{x})`, storing the output in ``out``. If ``length(x)`` is a
    compile-time constant, it is highly advised to explicitly pass in a value for ``chunk``.
    If ``chunk`` is not provided, ForwardDiff will try to guess an appropriate value based
    on ``x`` (see `the chunk configuration documentation <chunk.html>`__). If ``multithread =
    true``, then leverage multithreading when performing the computation. Note that
    multithreading support is experimental and may result in slowdown in some cases,
    especially for small input dimensions.

.. function:: ForwardDiff.gradient!(out::GradientResult, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false)

    Compute :math:`\nabla f(\vec{x})`. The value of :math:`f(\vec{x})` can then be accessed
    by calling ``ForwardDiff.value(out)``, while :math:`\nabla f(\vec{x})` can be accessed
    by calling ``ForwardDiff.gradient(out)``.

.. function:: ForwardDiff.gradient(f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false)

    Compute and return :math:`\nabla f(\vec{x})`.

Jacobians of :math:`f(x) : \mathbb{R}^n \to \mathbb{R}^m`
---------------------------------------------------------

.. function:: ForwardDiff.jacobian!(out, f, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, storing the output in ``out``. If ``length(x)``
    is a compile-time constant, it is highly advised to explicitly pass in a value for
    ``chunk``. If ``chunk`` is not provided, ForwardDiff will try to guess an appropriate
    value based on ``x`` (see `the chunk configuration documentation <chunk.html>`__).

.. function:: ForwardDiff.jacobian!(out, f!, y, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be called as
    ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``. The output
    matrix is stored in ``out``.

.. function:: ForwardDiff.jacobian!(out::JacobianResult, f!, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be called as
    ``f!(ForwardDiff.value(out), x)`` such that the output of :math:`f(\vec{x})` is stored
    in ``ForwardDiff.value(out)``. The output matrix can then be accessed by calling
    ``ForwardDiff.jacobian(out)``.

.. function:: ForwardDiff.jacobian(f, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`.

.. function:: ForwardDiff.jacobian(f!, y, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be
    called as ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``.

Hessians of :math:`f(x) : \mathbb{R}^n \to \mathbb{R}`
------------------------------------------------------

.. function:: ForwardDiff.hessian!(out, f, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute :math:`\mathbf{H}(f)(\vec{x})`, storing the output in ``out``. If ``length(x)``
    is a compile-time constant, it is highly advised to explicitly pass in a value for
    ``chunk``. If ``chunk`` is not provided, ForwardDiff will try to guess an appropriate
    value based on ``x`` (see `the chunk configuration documentation <chunk.html>`__).

.. function:: ForwardDiff.hessian!(out::HessianResult, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false)

    Compute :math:`\mathbf{H}(f)(\vec{x})`. The value of :math:`f(\vec{x})` can then be
    accessed by calling ``ForwardDiff.value(out)``, :math:`\nabla f(\vec{x})` can be
    accessed by calling ``ForwardDiff.gradient(out)``, and :math:`\mathbf{H}(f)(\vec{x})`
    can be accessed by calling ``ForwardDiff.hessian(out)``.

.. function:: ForwardDiff.hessian(f, x, chunk::Chunk = ForwardDiff.pickchunk(x))

    Compute and return :math:`\mathbf{H}(f)(\vec{x})`.
