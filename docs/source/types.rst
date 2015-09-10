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
        GradientNumber(value, partials::Partials) = new(value, partials)
        GradientNumber(value, partials::Tuple) = new(value, Partials(partials))
        GradientNumber(value, partials::Vector) = new(value, Partials(partials))
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
                    a
                    |
    +-------+-------+--------------+
    |       |       |              |   
   b₁ϵ     b₂ϵ     b₃ϵ     ...   b_N⋅ϵ

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
                                         a
                                         |                 
       +-----------+---------------------+----------------------------+  
       |           |                     |                            |  
      b₁ϵ₁        b₂ϵ₁                  b₃ϵ₁                         b₄ϵ₁
       |           |                     |                            |
       +         +---+               +---+---+                +---+-------+---+
       |        /     \             /    |    \              /    |       |    \
      b₁ϵ₂    b₁ϵ₂    b₂ϵ₂        b₁ϵ₂  b₂ϵ₂  b₃ϵ₂         b₁ϵ₂  b₂ϵ₂    b₃ϵ₂  b₄ϵ₂
       |       |       |          /      |      \          /      |       |      \
       +       +       +         +       +       +        +       +       +       +
       |       |       |        /        |        \       |       |       |       |
     c₁ϵ₁ϵ₂  c₂ϵ₁ϵ₂  c₃ϵ₁ϵ₂  c₄ϵ₁ϵ₂    c₅ϵ₁ϵ₂   c₆ϵ₁ϵ₂  c₇ϵ₁ϵ₂  c₈ϵ₁ϵ₂  c₉ϵ₁ϵ₂  c₉ϵ₁ϵ₂

Each root-to-leaf path represents an individual hyper-dual number. Labeling the indices of :math:`b` values on the second level with :math:`i`, and those on the third level with :math:`j`, the definition for an individual hyper-dual number :math:`h_{ij}` (where :math:`i \geq j`) in the above tree is:

.. math::
    
    h_{ij} = a + b_i \epsilon_1 + b_j \epsilon_2 + c_{q_{ij}} \epsilon_1 \epsilon_2 

where the :math:`c` indices are defined as:

.. math::

    q_{ij} = \frac{i(i - 1)}{2} + j

The algebra of hyper-dual numbers produces the following result for a function evaluation on an individual hyper-dual number:

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


In general, :math:`M = \frac{N(N+1)}{2}` individual hyper-dual numbers are stored by a ``HessianNumber{N}``. "Why :math:`M`?", you might ask. The answer is that ``h.hess`` (the :math:`c` value storage layer) is where :math:`2^{\text{nd}}`-order derivative information is accumulated, and we only need to take :math:`M` :math:`2^{\text{nd}}`-order derivatives to fully evaluate a symmetric :math:`N \times N` Hessian. 

``TensorNumber{N,T,C}``
-----------------------

.. code-block:: julia

    immutable TensorNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
        hessnum::HessianNumber{N,T,C}
        tens::Vector{T}
    end
