Basic ForwardDiff API
=====================

Derivatives of :math:`f(x) : \mathbb{R} \to \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k}`
--------------------------------------------------------------------------------------------------

Use ``ForwardDiff.derivative`` to differentiate functions of the form ``f(::Real...)::Real`` and ``f(::Real...)::AbstractArray``.

.. function:: ForwardDiff.derivative!(out, f, x)

    Compute :math:`f'(x)`, storing the output in ``out``.

.. function:: ForwardDiff.derivative!(out, f!, y, x, cfg = ForwardDiff.DerivativeConfig(f!, y, x))

    Compute and return :math:`f'(x)`, storing the output in ```out``. This form assumes that
    :math:`f'(x)` can be called as ``f!(y, x)`` such that the value result is stored in
    ``y``.

.. function:: ForwardDiff.derivative(f, x)

    Compute and return :math:`f'(x)`.

.. function:: ForwardDiff.derivative(f!, y, x, cfg = ForwardDiff.DerivativeConfig(f!, y, x))

    Compute and return :math:`f'(x)`. This form assumes that :math:`f'(x)` can be called as
    ``f!(y, x)`` such that the value result is stored in ``y``.

Gradients of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}`
------------------------------------------------------------------------------------------------

Use ``ForwardDiff.gradient`` to differentiate functions of the form ``f(::AbstractArray)::Real``.

.. function:: ForwardDiff.gradient!(out, f, x, cfg = ForwardDiff.GradientConfig(f, x))

    Compute :math:`\nabla f(\vec{x})`, storing the output in ``out``. It is highly advised
    to preallocate ``cfg`` yourself (see the `AbstractConfig
    <basic_api.html#the-abstractconfig-types>`_ section below).

.. function:: ForwardDiff.gradient(f, x, cfg = ForwardDiff.GradientConfig(x))

    Compute and return :math:`\nabla f(\vec{x})`.

Jacobians of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}^{m_1} \times \dots \times \mathbb{R}^{m_k}`
-------------------------------------------------------------------------------------------------------------------------------------------

Use ``ForwardDiff.jacobian`` to differentiate functions of the form ``f(::AbstractArray)::AbstractArray``.

.. function:: ForwardDiff.jacobian!(out, f, x, cfg = ForwardDiff.JacobianConfig(f, x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, storing the output in ``out``. It is highly
    advised to preallocate ``cfg`` yourself (see the `AbstractConfig
    <basic_api.html#the-abstractconfig-types>`_ section below).

.. function:: ForwardDiff.jacobian!(out, f!, y, x, cfg = ForwardDiff.JacobianConfig(f!, y, x))

    Compute :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be called as
    ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``. The output
    matrix is stored in ``out``.

.. function:: ForwardDiff.jacobian(f, x, cfg = ForwardDiff.JacobianConfig(f, x))

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`.

.. function:: ForwardDiff.jacobian(f!, y, x, cfg = ForwardDiff.JacobianConfig(f!, y, x))

    Compute and return :math:`\mathbf{J}(f)(\vec{x})`, where :math:`f(\vec{x})` can be
    called as ``f!(y, x)`` such that the output of :math:`f(\vec{x})` is stored in ``y``.

Hessians of :math:`f(x) : \mathbb{R}^{n_1} \times \dots \times \mathbb{R}^{n_k} \to \mathbb{R}`
-----------------------------------------------------------------------------------------------

Use ``ForwardDiff.hessian`` to perform second-order differentiation on functions of the form ``f(::AbstractArray)::Real``.

.. function:: ForwardDiff.hessian!(out, f, x, cfg = ForwardDiff.HessianConfig(f, x))

    Compute :math:`\mathbf{H}(f)(\vec{x})`, storing the output in ``out``. It is highly
    advised to preallocate ``cfg`` yourself (see the `AbstractConfig
    <basic_api.html#the-abstractconfig-types>`_ section below).

.. function:: ForwardDiff.hessian(f, x, cfg = ForwardDiff.HessianConfig(f, x))

    Compute and return :math:`\mathbf{H}(f)(\vec{x})`.

The ``AbstractConfig`` Types
----------------------------

For the sake of convenience and performance, all "extra" information used by ForwardDiff's
API methods is bundled up in the ``ForwardDiff.AbstractConfig`` family of types. Theses
types allow the user to easily feed several different parameters to ForwardDiff's  API
methods, such as `chunk size <advanced_usage.html#configuring-chunk-size>`_, work buffers,
and perturbation seed configurations.

ForwardDiff's basic API methods will allocate these types automatically by default,
but you can drastically reduce memory usage if you preallocate them yourself.

Note that for all constructors below, the chunk size ``N`` may be explictly provided,
or omitted, in which case ForwardDiff will automatically select a chunk size for you.
However, it is highly recomended to `specify the chunk size manually when possible
<advanced_usage.html#configuring-chunk-size>`_.

Note also that configurations constructed for a specific function ``f`` cannot
be reused to differentiate other functions (though can be reused to differentiate
``f`` at different values). To construct a configuration which can be reused to
differentiate any function, you can pass ``nothing`` as the function argument.
While this is more flexible, this decreases ForwardDiff's ability to catch
and prevent `perturbation confusion`_.

.. function:: ForwardDiff.GradientConfig(f, x, chunk::ForwardDiff.Chunk{N} = Chunk(x))

    Construct a ``GradientConfig`` instance based on the type of ``f`` and
    type/shape of the input vector ``x``. The returned ``GradientConfig``
    instance contains all the work buffers required by ForwardDiff's gradient
    methods.

    This constructor does not store/modify ``x``.

.. function:: ForwardDiff.JacobianConfig(f, x, chunk::ForwardDiff.Chunk{N} = Chunk(x))

    Exactly like the ``GradientConfig`` constructor, but returns a ``JacobianConfig`` instead.

.. function:: ForwardDiff.JacobianConfig(f!, y, x, chunk::ForwardDiff.Chunk{N} = Chunk(x))

    Construct a ``JacobianConfig`` instance based on the type of ``f!``, and the
    types/shapes of the output vector ``y`` and the input vector ``x``. The
    returned ``JacobianConfig`` instance contains all the work buffers required
    by ``ForwardDiff.jacobian``/``ForwardDiff.jacobian!`` when the target
    function takes the form ``f!(y, x)``.

    This constructor does not store/modify ``y`` or ``x``.

.. function:: ForwardDiff.HessianConfig(f, x, chunk::ForwardDiff.Chunk{N} = Chunk(x))

    Construct a ``HessianConfig`` instance based on the type of ``f`` and
    type/shape of the input vector ``x``. The returned ``HessianConfig`` instance contains
    all the work buffers required by ForwardDiff's Hessian methods. If using
    ``ForwardDiff.hessian!(out::DiffBase.DiffResult, f, x)``, use the constructor
    ``ForwardDiff.HessianConfig(f, out, x, chunk)`` instead.

    This constructor does not store/modify ``x``.

.. function:: ForwardDiff.HessianConfig(f, out::DiffBase.DiffResult, x, chunk::ForwardDiff.Chunk{N} = Chunk(x))

    Construct an ``HessianConfig`` instance based on the type of ``f``, types/storage
    in ``out``, and type/shape of the input vector ``x``. The returned ``HessianConfig``
    instance contains all the work buffers required by
    ``ForwardDiff.hessian!(out::DiffBase.DiffResult, args...)``.

    This constructor does not store/modify ``out`` or ``x``.

.. _`perturbation confusion`: https://github.com/JuliaDiff/ForwardDiff.jl/issues/83
