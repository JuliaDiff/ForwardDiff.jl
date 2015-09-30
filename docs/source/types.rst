``ForwardDiffNumber`` Types
===========================

You should familiarize yourself with `dual numbers`_ and `hyper-dual numbers`_ before reading this document.

.. _`dual numbers`: https://en.wikipedia.org/wiki/Dual_number
.. _`hyper-dual numbers`: https://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf

``Partials{T,C}``
-----------------

.. code-block:: julia

    immutable Partials{T,C}
        data::C
        Partials{N}(data::NTuple{N,T}) = new(data)
        Partials(data::Vector{T}) = new(data)
    end

The ``Partials`` type serves as an abstraction over the container type used to store partial derivatives. As you can see, both ``Tuple`` and ``Vector`` types are supported.

When the number of partial derivatives being stored is small (~10), ForwardDiff stores them in a ``Tuple``. ``Tuple`` elements are stack-allocated, potentially allowing for very fast operations with little GC overhead.

When the number of partial derivatives being stored is large enough (greater than ~10), ForwardDiff elects to store them in a heap-allocated ``Vector``. While operations on vectors can be slower (depending on access patterns), large vectors will not "clog" or overflow the stack, as large tuples would.

``GradientNumber{N,T,C}``
-------------------------

.. code-block:: julia

    immutable GradientNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
        value::T
        partials::Partials{T,C}
    end


The ``GradientNumber`` type represents a "generalized" dual number. Instead of having a single :math:`\epsilon`-component, a generalized dual number has :math:`N` :math:`\epsilon`-components:

.. math::

    g_N = a + \sum_{i=1}^N b_i \epsilon_i

    f(g_N) = f(a) + f'(a) \sum_{i=1}^N b_i \epsilon_i

Relating this equation to the above type definition, we have :math:`a \to` ``g.value`` and :math:`b_i \to` ``g.partials[i]``. Here's what it might look like to overload Julia's ``sin`` function on a ``GradientNumber``:

.. code-block:: julia

    function Base.sin(g::GradientNumber)
        a = g.value
        v, deriv = sin(a), cos(a)
        return GradientNumber(v, deriv*g.partials)
    end

The actual definition utilizes promotion rules and stricter typing, but you get the idea.

By setting the appropriate :math:`b` values (i.e. the ``partials`` field), we can construct vectors like :math:`\vec{x}_{g_N}` below. Passing :math:`\vec{x}_{g_N}` to a given function :math:`f: \mathbb{R}^N \to \mathbb{R}^N` produces :math:`f`'s directional derivatives evaluated at the original input :math:`\vec{x}`:

.. math::

    \vec{x} =
    \begin{bmatrix}
    x_1 \\
    \vdots \\
    x_i \\
    \vdots \\
    x_N
    \end{bmatrix} \to
    \vec{x}_{g_N} =
    \begin{bmatrix}
    x_1 + \epsilon_1 \\
    \vdots \\
    x_i + \epsilon_i \\
    \vdots \\
    x_N + \epsilon_N
    \end{bmatrix} \to
    f(\vec{x}_{g_N}) =
    f(\vec{x}) + \sum_{i=1}^N \frac{\delta f(\vec{x})}{\delta x_i} \epsilon_i

Alternatively, the ``GradientNumber`` type can be interpreted as an ensemble of :math:`N` dual numbers, all sharing the same real component. This interpretation will be the easier one to discuss when we examine the ``HessianNumber`` and ``TensorNumber`` types.

The tree below visualizes this interpretation. The first level of the tree is given by ``g.value``, while the second level is comprised of the elements of ``g.partials``:

.. code-block:: none

    g =

       /------------- a
      /             / | \
     +       +-----/  +  \-----------+
     |       |        |              |
    b₁ϵ     b₂ϵ      b₃ϵ     ...   b_N⋅ϵ

``HessianNumber{N,T,C}``
------------------------

.. code-block:: julia

    immutable HessianNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
        gradnum::GradientNumber{N,T,C}
        hess::Vector{T}
    end

A ``HessianNumber`` can be intepreted as an ensemble of hyper-dual numbers. The representation of an instance of ``HessianNumber{4}`` is visualized by the tree below. The :math:`a` and :math:`b` values are stored in ``h.gradnum``, while the :math:`c` values are the elements of ``h.hess``:

.. code-block:: none

    h =
        /------------------------------ a
       /                              / | \
      +           +------------------/  +  \------------------------+
      |           |                     |                           |
     b₁ϵ₁        b₂ϵ₁                  b₃ϵ₁                    --- b₄ϵ₁----
      |          / \                 /  |  \                  /   /    \   \
      +         +   +               +   +   +                +   +      +   +
      |        /     \             /    |    \              /    |      |    \
     b₁ϵ₂    b₁ϵ₂    b₂ϵ₂        b₁ϵ₂  b₂ϵ₂  b₃ϵ₂         b₁ϵ₂  b₂ϵ₂   b₃ϵ₂  b₄ϵ₂
      |       |       |          /      |      \          /      |      |      \
      +       +       +         +       +       +        +       +      +       +
      |       |       |        /        |        \       |       |      |       |
    c₁ϵ₁ϵ₂  c₂ϵ₁ϵ₂  c₃ϵ₁ϵ₂  c₄ϵ₁ϵ₂    c₅ϵ₁ϵ₂   c₆ϵ₁ϵ₂  c₇ϵ₁ϵ₂  c₈ϵ₁ϵ₂  c₉ϵ₁ϵ₂  c₉ϵ₁ϵ₂

Each root-to-leaf path represents an individual hyper-dual number. Labeling the indices of :math:`b` values on the :math:`\epsilon_1` level with :math:`i`, and those on the :math:`\epsilon_2` level with :math:`j`, the definition for an individual hyper-dual number :math:`h_{ij}` (where :math:`i \geq j`) in the above tree is:

.. math::

    h_{ij} = a + b_i \epsilon_1 + b_j \epsilon_2 + c_{q_{ij}} \epsilon_1 \epsilon_2

where the :math:`c` indices are defined as:

.. math::

    q_{ij} = \frac{i(i - 1)}{2} + j

The following defines a univariate function evaluation on an individual hyper-dual number:

.. math::

    f(h_{ij}) = f(a) + f'(a) b_i \epsilon_1
                     + f'(a) b_j \epsilon_2
                     + (f'(a) c_{q_{ij}} + f''(a) b_i b_j) \epsilon_1 \epsilon_2

To illustrate, here's the definition for the :math:`4^{\text{th}}` hyper-dual number stored in a ``HessianNumber``:

.. math::

    h_4 = a + b_2 \epsilon_1 + b_3 \epsilon_2 + c_4 \epsilon_1 \epsilon_2

    f(h_4) = f(a) + f'(a) b_2 \epsilon_1
                  + f'(a) b_3 \epsilon_2
                  + (f'(a) c_4 + f''(a) b_2 b_3) \epsilon_1 \epsilon_2


In general, :math:`M = \frac{N(N+1)}{2}` individual hyper-dual numbers are stored by a ``HessianNumber{N}``. This because that we only need to take :math:`M` :math:`2^{\text{nd}}`-order derivatives to fully evaluate a symmetric :math:`N \times N` Hessian.

``TensorNumber{N,T,C}``
-----------------------

.. note::

    AD folks tend to abuse the word "tensor"; in this context, it refers to a :math:`3^{\text{rd}}` order generalization of the Hessian. Given a function :math:`f:\mathbb{R}^N \to \mathbb{R}`, the tensor operator :math:`\mathbf{T}` is defined as

    .. math::

        \mathbf{T}(f) = \sum_{i,j,k=1}^{N} \frac{\delta^3 f}{\delta x_i \delta x_j \delta x_k}


.. code-block:: julia

    immutable TensorNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
        hessnum::HessianNumber{N,T,C}
        tens::Vector{T}
    end

The ``TensorNumber`` type is essentially the same as the ``HessianNumber`` type, but with a third :math:`\epsilon` component that allows for :math:`3^{\text{rd}}`-order derivative accumulation. The paper by Fike and Alonso describing hyper-dual numbers mentions this :math:`3^{\text{rd}}`-order variation, but doesn't go into too much detail. If you work it out yourself, you find that a :math:`3^{\text{rd}}`-order hyper-dual number essentially looks like this:

.. math::

    t = t_0 + t_1 \epsilon_1 + t_2 \epsilon_2 + t_3 \epsilon_3 + t_4 \epsilon_1 \epsilon_2 + t_5 \epsilon_1 \epsilon_3 + t_6 \epsilon_2 \epsilon_3 + t_7 \epsilon_1 \epsilon_2 \epsilon_3

The ``TensorNumber`` type stores an ensemble of such numbers. The  type's ``tens`` field is used to store the coefficients of the :math:`\epsilon_1 \epsilon_2 \epsilon_3` terms (:math:`t_7` above). All the other components are contained in the ``hessnum`` field.

The action of a univariate function :math:`f` on an individual :math:`3^{\text{rd}}`-order hyper-dual number is defined as:

.. math::

    f(t) = & f(t_0) + f'(t_0) \cdot (t_1 ϵ_1 + t_2 ϵ_2 + t_3 ϵ_3 + t_4 ϵ_1 ϵ_2 + t_5 ϵ_1 ϵ_3 + t_6 ϵ_2 ϵ_3 + t_7 ϵ_1 ϵ_2 ϵ_3) + \\
           & f''(t_0) \cdot (t_1 t_2 ϵ_2 ϵ_1 + t_1 t_3 ϵ_3 ϵ₁ + t_3 t_4 ϵ_2 ϵ_3 ϵ_1 + t_2 t_5 ϵ_2 ϵ_3 ϵ_1 + t_1 t_6 ϵ_2 ϵ_3 ϵ_1 + t_2 t_3 ϵ_2 ϵ_3) + \\
           & f'''(t_0) \cdot (t_1 t_2 t_3 ϵ_2 ϵ_3 ϵ_1)

We can rearrange this to get the new coefficient of the :math:`\epsilon_1 \epsilon_2 \epsilon_3` term:


.. math::

    f(t)_{\epsilon_1 \epsilon_2 \epsilon_3} = f'(t_0) \cdot t_7 + f''(t_0) \cdot (t_3 t_4 + t_2 t_5 + t_1 t_6) + f'''(t_0) \cdot t_1 t_2 t_3


As was said, the ``TensorNumber`` type is basically a :math:`3^{\text{rd}}`-order extension of the previous types - it's an ensemble of hyper-dual numbers, and is implemented such that the lower-order partial values are reused where possible. Thus, one could draw the same kind of tree representation for a ``TensorNumber`` instance that is drawn above for the ``HessianNumber`` and ``GradientNumber`` types (though even for :math:`N=4`, it's too large for us to show here).

Relating the mathematics to the implementation, here's what the indexing structure looks like for an individual number in the ``TensorNumber`` ensemble (where the :math:`a`, :math:`b`, and :math:`c` values are stored in ``t.hessnum``, while the :math:`d` values are stored in ``t.tens``):

.. math::

    t_{N_{ijk}} = a + b_i \epsilon_1 + b_j \epsilon_2 + b_k \epsilon_3 + c_{q_{ij}} \epsilon_1 \epsilon_2 + c_{q_{ik}} \epsilon_1 \epsilon_3 + c_{q_{jk}} \epsilon_2 \epsilon_3 + d_{p_{ijkN}} \epsilon_1 \epsilon_2 \epsilon_3

where

.. math::

    1 \leq i \leq N

    i \leq j \leq N

    i \leq k \leq j

    M = \frac{N(N+1)(N+2)}{6}

    d_p \in \{d_1, d_2, d_3...d_{M-1}, d_M\}

    p_{ijkN} &= \left[\sum_{\alpha=1}^{i-1} \sum_{\beta=\alpha}^{N} \sum_{\gamma=\alpha}^{\beta} 1 \right] + \left[\sum_{\alpha=i}^{j-1} \sum_{\beta=i}^{\alpha} 1 \right] + (k - i + 1) \\

This rather complex indexing structure is derived from the loop code written to taken advantage of the tensor's tri-fold symmetry. A given tensor :math:`\mathbf{T}(f)` is generally symmetric under index order permutation:

.. math::

    \mathbf{T}(f)_{ijk} = \mathbf{T}(f)_{ikj} = \mathbf{T}(f)_{jki} = \mathbf{T}(f)_{jik} = \mathbf{T}(f)_{kij} = \mathbf{T}(f)_{kji}
