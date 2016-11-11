How ForwardDiff Works
=====================

ForwardDiff is an implementation of `forward mode automatic differentiation`_ (AD) in
Julia. There are two key components of this implementation: the ``Dual`` type, and the API.

.. _`forward mode automatic differentiation`: https://en.wikipedia.org/wiki/Automatic_differentiation

Dual Number Implementation
--------------------------

Partial derivatives are stored in the ``Partials{N,T}`` type:

.. code-block:: julia

    immutable Partials{N,T}
        values::NTuple{N,T}
    end

Overtop of this container type, ForwardDiff implements the ``Dual{N,T}`` type:

.. code-block:: julia

    immutable Dual{N,T<:Real} <: Real
        value::T
        partials::Partials{N,T}
    end

This type represents an ``N``-dimensional `dual number`_ with the following mathematical
behavior:

.. math::

    f(a + \sum_{i=1}^N b_i \epsilon_i) = f(a) + f'(a) \sum_{i=1}^N b_i \epsilon_i

where the :math:`a` component is stored in the ``value`` field and the :math:`b`
components are stored in the ``partials`` field. This property of dual numbers is the
central feature that allows ForwardDiff to take derivatives.

In order to implement the above property, elementary numerical functions on a ``Dual``
number are overloaded to evaluate both the original function, *and* evaluate the derivative
of the function, propogating the derivative via multiplication. For example, ``Base.sin``
can be overloaded on ``Dual`` like so:

.. code-block:: julia

    Base.sin(d::Dual) = Dual(sin(value(d)), cos(value(d)) * partials(d))

If we assume that a general function ``f`` is composed of entirely of these elementary
functions, then the chain rule enables our derivatives to compose as well. Thus, by
overloading a plethora of elementary functions, we can differentiate generic functions
composed of them by passing in a ``Dual`` number and looking at the output.

We won't dicuss higher-order differentiation in detail, but the reader is encouraged to
learn about `hyper-dual numbers`_, which extend dual numbers to higher orders by introducing
extra :math:`\epsilon` terms that can cross-multiply. ForwardDiff's ``Dual`` number
implementation naturally supports hyper-dual numbers without additional code by allowing
instances of the ``Dual`` type to nest within each other. For example, a second-order
hyper-dual number has the type ``Dual{N,Dual{N,T}}``, a third-order hyper-dual number has
the type ``Dual{N,Dual{N,Dual{N,T}}}``, and so on.

.. _`dual number`: https://en.wikipedia.org/wiki/Dual_number
.. _`hyper-dual numbers`: https://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf

ForwardDiff's API
-----------------

The second component provided by this package is the API, which abstracts away the number
types and makes it easy to execute familiar calculations like gradients and Hessians. This
way, users don't have to understand ``Dual`` numbers in order to make use of the package.

The job of the API functions is to performantly seed input values with ``Dual`` numbers,
pass the seeded value into the target function, and extract the derivative information from
the result. For example, to calculate the partial derivatives for the gradient of a function
:math:`f` at an input vector :math:`\vec{x}`, we would do the following:

.. math::

    \vec{x} = \begin{bmatrix}
                   x_1 \\
                   \vdots \\
                   x_i \\
                   \vdots \\
                   x_N
               \end{bmatrix}
    \to
    \vec{x}_{\epsilon} = \begin{bmatrix}
                             x_1 + \epsilon_1 \\
                             \vdots \\
                             x_i + \epsilon_i \\
                             \vdots \\
                             x_N + \epsilon_N
                         \end{bmatrix}
    \to
    f(\vec{x}_{\epsilon}) = f(\vec{x}) + \sum_{i=1}^N \frac{\delta f(\vec{x})}{\delta x_i} \epsilon_i

In reality, ForwardDiff does this calculation in `chunks of the input vector
<advanced_usage.html#configuring-chunk-size>`_. To provide a simple example of this, let's
examine the case where the input vector size is 4 and the chunk size is 2. It then takes two
calls to :math:`f` to evaluate the gradient:

.. math::

    \vec{x} = \begin{bmatrix}
                   x_1 \\
                   x_2 \\
                   x_3 \\
                   x_4
               \end{bmatrix}

    \vec{x}_{\epsilon} = \begin{bmatrix}
                            x_1 + \epsilon_1 \\
                            x_2 + \epsilon_2 \\
                            x_3 \\
                            x_4
                         \end{bmatrix}
    \to
    f(\vec{x}_{\epsilon}) = f(\vec{x}) + \frac{\delta f(\vec{x})}{\delta x_1} \epsilon_1 + \frac{\delta f(\vec{x})}{\delta x_2} \epsilon_2

    \vec{x}_{\epsilon} = \begin{bmatrix}
                            x_1 \\
                            x_2 \\
                            x_3 + \epsilon_1 \\
                            x_4 + \epsilon_2
                         \end{bmatrix}
    \to
    f(\vec{x}_{\epsilon}) = f(\vec{x}) + \frac{\delta f(\vec{x})}{\delta x_3} \epsilon_1 + \frac{\delta f(\vec{x})}{\delta x_4} \epsilon_2

This seeding process is similar for Jacobians, so we won't rehash it here.
