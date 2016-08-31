ForwardDiff API
===============

Derivatives of :math:`f(x) : \mathbb{R} \to \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k}`
--------------------------------------------------------------------------------------------------

Use ``ForwardDiff.derivative`` to differentiate functions of the form ``f(::Real)::Real`` and ``f(::Real)::AbstractArray``.

.. function:: ForwardDiff.derivative!(out, f, x)

    Compute :math:`f'(x)`, storing the output in ``out``.

.. function:: ForwardDiff.derivative!(out::DerivativeResult, f, x)

    Compute :math:`f'(x)`, storing the output in ``out``. The value of :math:`f(x)` can be
    accessed by calling ``ForwardDiff.value(out)``, while :math:`f'(x)` can be accessed by
    calling ``ForwardDiff.derivative(out)``.

.. function:: ForwardDiff.derivative(f, x)

    Compute and return :math:`f'(x)`.

Gradients of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}`
------------------------------------------------------------------------------------------------

Use ``ForwardDiff.gradient`` to differentiate functions of the form ``f(::AbstractArray)::Real``.

.. function:: ForwardDiff.gradient!(out, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false, usecache = true)

    Compute :math:`\nabla f(\vec{x})`, storing the output in ``out``.

    If ``length(x)`` is a compile-time constant, it is highly advised to explicitly pass in
    a value for ``chunk``. If ``chunk`` is not provided, ForwardDiff will try to guess an
    appropriate value based on ``x`` (see `the chunk configuration documentation
    <chunk.html>`__).

    If ``multithread = true``, then leverage multithreading when performing the computation.
    Note that multithreading support is experimental and may result in slowdown in some
    cases, especially for small input dimensions.

    If ``usecache = false``, then don't rely on ForwardDiff's cache to store/retrieve
    internally used work arrays. If ``length(x)`` is very small, or you need your
    calculation to be task-safe, then you might benefit from setting ``usecache = false``.

.. function:: ForwardDiff.gradient!(out::GradientResult, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false, usecache = true)

    Compute :math:`\nabla f(\vec{x})`. The value of :math:`f(\vec{x})` can then be accessed
    by calling ``ForwardDiff.value(out)``, while :math:`\nabla f(\vec{x})` can be accessed
    by calling ``ForwardDiff.gradient(out)``.

.. function:: ForwardDiff.gradient(f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false, usecache = true)

    Compute and return :math:`\nabla f(\vec{x})`.

Jacobians of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}^{m_1} \times \dots \times \mathbb{R}^{m_k}`
-------------------------------------------------------------------------------------------------------------------------------------------

Use ``ForwardDiff.jacobian`` to differentiate functions of the form ``f(::AbstractArray)::AbstractArray``.

.. function:: ForwardDiff.jacobian!(out, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); usecache = true)

    Compute :math:`\mathbf{J}(f)(\vec{x})`, storing the output in ``out``.

    If ``length(x)`` is a compile-time constant, it is highly advised to explicitly pass in
    a value for ``chunk``. If ``chunk`` is not provided, ForwardDiff will try to guess an
    appropriate value based on ``x`` (see `the chunk configuration documentation
    <chunk.html>`__).

    If ``usecache = false``, then don't rely on ForwardDiff's cache to store/retrieve
    internally used work arrays. If ``length(x)`` is very small, or you need your
    calculation to be task-safe, then you might benefit from setting ``usecache = false``.

.. function:: ForwardDiff.jacobian!(out, f!, y, x, chunk::Chunk = ForwardDiff.pickchunk(x); usecache = true)

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be called as
    ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``. The output
    matrix is stored in ``out``.

.. function:: ForwardDiff.jacobian!(out::JacobianResult, f!, x, chunk::Chunk = ForwardDiff.pickchunk(x); usecache = true)

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be called as
    ``f!(ForwardDiff.value(out), x)`` such that the output of :math:`f(\vec{x})` is stored
    in ``ForwardDiff.value(out)``. The output matrix can then be accessed by calling
    ``ForwardDiff.jacobian(out)``.

.. function:: ForwardDiff.jacobian(f, x, chunk::Chunk = ForwardDiff.pickchunk(x); usecache = true)

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`.

.. function:: ForwardDiff.jacobian(f!, y, x, chunk::Chunk = ForwardDiff.pickchunk(x); usecache = true)

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be
    called as ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``.

Hessians of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}`
-----------------------------------------------------------------------------------------------

Use ``ForwardDiff.hessian`` to perform second-order differentiation on functions of the form ``f(::AbstractArray)::Real``.

.. function:: ForwardDiff.hessian!(out, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false, usecache = true)

    Compute :math:`\mathbf{H}(f)(\vec{x})`, storing the output in ``out``.

    If ``length(x)`` is a compile-time constant, it is highly advised to explicitly pass in
    a value for ``chunk``. If ``chunk`` is not provided, ForwardDiff will try to guess an
    appropriate value based on ``x`` (see `the chunk configuration documentation
    <chunk.html>`__).

    If ``multithread = true``, then leverage multithreading when performing the computation.
    Note that multithreading support is experimental and may result in slowdown in some
    cases, especially for small input dimensions.

    If ``usecache = false``, then don't rely on ForwardDiff's cache to store/retrieve
    internally used work arrays. If ``length(x)`` is very small, or you need your
    calculation to be task-safe, then you might benefit from setting ``usecache = false``.

.. function:: ForwardDiff.hessian!(out::HessianResult, f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false, usecache = true)

    Compute :math:`\mathbf{H}(f)(\vec{x})`. The value of :math:`f(\vec{x})` can then be
    accessed by calling ``ForwardDiff.value(out)``, :math:`\nabla f(\vec{x})` can be
    accessed by calling ``ForwardDiff.gradient(out)``, and :math:`\mathbf{H}(f)(\vec{x})`
    can be accessed by calling ``ForwardDiff.hessian(out)``.

.. function:: ForwardDiff.hessian(f, x, chunk::Chunk = ForwardDiff.pickchunk(x); multithread = false, usecache = true)

    Compute and return :math:`\mathbf{H}(f)(\vec{x})`.

Hessian of a vector-valued function
-----------------------------------

While ForwardDiff does not have a built-in function for taking Hessians of vector-valued
functions, you can easily compose calls to ``ForwardDiff.jacobian`` to accomplish this.
For example:

.. code-block:: julia

    julia> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(sin, x), [1,2,3])
    9×3 Array{Float64,2}:
     -0.841471   0.0        0.0
     -0.0       -0.0       -0.0
     -0.0       -0.0       -0.0
     0.0        0.0        0.0
     -0.0       -0.909297  -0.0
     -0.0       -0.0       -0.0
     0.0        0.0        0.0
     -0.0       -0.0       -0.0
     -0.0       -0.0       -0.14112

Since this functionality is composed from ForwardDiff's existing API rather than built into
it, you're free to construct a ``vector_hessian`` function which suits your needs. For
example, if you require the shape of the output to be a tensor rather than a block matrix,
you can do so with a ``reshape`` (note that ``reshape`` does not copy data, so it's not an
expensive operation):

.. code-block:: julia

    julia> function vector_hessian(f, x)
           n = length(x)
           out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
           return reshape(out, n, n, n)
       end
    vector_hessian (generic function with 1 method)

    julia> vector_hessian(sin, [1, 2, 3])
    3×3×3 Array{Float64,3}:
    [:, :, 1] =
     -0.841471   0.0   0.0
     -0.0       -0.0  -0.0
     -0.0       -0.0  -0.0

    [:, :, 2] =
      0.0   0.0        0.0
     -0.0  -0.909297  -0.0
     -0.0  -0.0       -0.0

    [:, :, 3] =
      0.0   0.0   0.0
     -0.0  -0.0  -0.0
     -0.0  -0.0  -0.14112

Likewise, you could write a version of ``vector_hessian`` which supports functions of the
form ``f!(y, x)``, or perhaps an in-place Jacobian with ``ForwardDiff.jacobian!``.
