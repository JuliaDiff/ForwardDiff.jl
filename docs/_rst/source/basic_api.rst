Basic ForwardDiff API
=====================

Derivatives of :math:`f(x) : \mathbb{R} \to \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k}`
--------------------------------------------------------------------------------------------------

Use ``ForwardDiff.derivative`` to differentiate functions of the form ``f(::Real)::Real`` and ``f(::Real)::AbstractArray``.

.. function:: ForwardDiff.derivative!(out, f, x)

    Compute :math:`f'(x)`, storing the output in ``out``.

.. function:: ForwardDiff.derivative(f, x)

    Compute and return :math:`f'(x)`.

Gradients of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}`
------------------------------------------------------------------------------------------------

Use ``ForwardDiff.gradient`` to differentiate functions of the form ``f(::AbstractArray)::Real``.

.. function:: ForwardDiff.gradient!(out, f, x, cfg = ForwardDiff.Config(x))

    Compute :math:`\nabla f(\vec{x})`, storing the output in ``out``. It is highly
    advised to preallocate ``cfg`` yourself (see the `Config`_ section below).

.. function:: ForwardDiff.gradient(f, x, cfg = ForwardDiff.Config(x))

    Compute and return :math:`\nabla f(\vec{x})`.

Jacobians of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}^{m_1} \times \dots \times \mathbb{R}^{m_k}`
-------------------------------------------------------------------------------------------------------------------------------------------

Use ``ForwardDiff.jacobian`` to differentiate functions of the form ``f(::AbstractArray)::AbstractArray``.

.. function:: ForwardDiff.jacobian!(out, f, x, cfg = ForwardDiff.Config(x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, storing the output in ``out``. It is highly
    advised to preallocate ``cfg`` yourself (see the `Config`_ section below).

.. function:: ForwardDiff.jacobian!(out, f!, y, x, cfg = ForwardDiff.Config(y, x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be called as
    ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``. The output
    matrix is stored in ``out``.

.. function:: ForwardDiff.jacobian(f, x, cfg = ForwardDiff.Config(x))

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`.

.. function:: ForwardDiff.jacobian(f!, y, x, cfg = ForwardDiff.Config(y, x))

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be
    called as ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``.

Hessians of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}`
-----------------------------------------------------------------------------------------------

Use ``ForwardDiff.hessian`` to perform second-order differentiation on functions of the form ``f(::AbstractArray)::Real``.

.. function:: ForwardDiff.hessian!(out, f, x, cfg = ForwardDiff.HessianConfig(x))

    Compute :math:`\mathbf{H}(f)(\vec{x})`, storing the output in ``out``. It is highly
    advised to preallocate ``cfg`` yourself (see the `Config`_ section below).

.. function:: ForwardDiff.hessian(f, x, cfg = ForwardDiff.HessianConfig(x))

    Compute and return :math:`\mathbf{H}(f)(\vec{x})`.

The ``Config`` Type
-------------------

For the sake of convenience and performance, all "extra" information used by ForwardDiff's
API methods is bundled up in the ``ForwardDiff.AbstractConfig`` family of types. Theses
types allow the user to easily feed several different parameters to ForwardDiff's  API
methods, such as `chunk size <advanced_usage.html#configuring-chunk-size>`_, work buffers,
multithreading configurations, and perturbation seed configurations.

ForwardDiff's basic API methods will allocate these types automatically by default,
but you can drastically reduce memory usage if you preallocate them yourself.

Note that for all constructors below, the chunk size ``N`` may be explictly provided as a
type parameter, or omitted, in which case ForwardDiff will automatically select a chunk size
for you. However, it is highly recomended to `specify the chunk size manually when possible
<advanced_usage.html#configuring-chunk-size>`_.

.. function:: ForwardDiff.Config{N}(x)

    Construct an ``Config`` instance based on the type and shape of the input vector ``x``.
    The returned ``Config`` instance contains all the work buffers required by
    ForwardDiff's gradient/Jacobian methods. If taking the Jacobian of a target function
    with the form ``f!(y, x)``, use the constructor ``ForwardDiff.Config{N}(y, x)``
    instead.

    This constructor does not store/modify ``x``.

.. function:: ForwardDiff.Config{N}(y, x)

    Construct an ``Config`` instance based on the type and shape of the output vector ``y``
    and the input vector ``x``. The returned ``Config`` instance contains all the work
    buffers required by  ``ForwardDiff.jacobian``/``ForwardDiff.jacobian!`` with a target
    function of the form ``f!(y, x)``.

    This constructor does not store/modify ``y`` or ``x``.

.. function:: ForwardDiff.HessianConfig{N}(x)

    Construct a ``HessianConfig`` instance based on the type and shape of the input vector
    ``x``. The returned ``HessianConfig`` instance contains all the work buffers required
    by ForwardDiff's Hessian methods. If using
    ``ForwardDiff.hessian!(out::DiffBase.DiffResult, args...)``, use the constructor
    ``ForwardDiff.HessianConfig{N}(out, x)`` instead.

    This constructor does not store/modify ``x``.

.. function:: ForwardDiff.HessianConfig{N}(out::DiffBase.DiffResult, x)

    Construct an ``HessianConfig`` instance based on the type and shape of the storage in
    ``out`` and the input vector ``x``. The returned ``HessianConfig`` instance contains
    all the work buffers required by ``ForwardDiff.hessian!(out::DiffBase.DiffResult,
    args...)``.

    This constructor does not store/modify ``out`` or ``x``.

.. function:: ForwardDiff.Multithread(cfg::AbstractConfig)

    Wrap the given ``cfg`` in a ``Multithread`` instance, which can then be passed to
    gradient or Hessian methods in order to enable experimental multithreading. Jacobian
    methods do not yet support multithreading.

    Note that multithreaded ForwardDiff API methods will attempt to use all available
    threads. In the future, once Julia exposes more fine-grained threading primitives,
    a ``Multithread`` constructor may be added which takes in a user-provided subset
    of thread IDs instead of using all available threads.
