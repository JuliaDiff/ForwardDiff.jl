How to Contribute
=================

There are a few fairly easy ways for newcomers to substantially improve ForwardDiff.jl, and they all revolve around **writing functions for the** ``ForwardDiffNumber`` **types**. This section provides brief tutorials on how to make these contributions.

If you're new GitHub, here's an outline of the workflow you should use:

1. Fork ForwardDiff.jl
2. Make a new branch on your fork, named after whatever changes you'll be making
3. Apply your code changes to the branch on your fork
4. When you're done, submit a PR to ForwardDiff.jl to merge your fork into ForwardDiff.jl's master branch.

Manually Optimizing Unary Functions
-----------------------------------

1. Pick a function to optimize
++++++++++++++++++++++++++++++

To see a list of functions to pick from, look at ``ForwardDiff.auto_defined_unary_funcs``:

.. code-block:: julia

    julia> import ForwardDiff

    julia> ForwardDiff.auto_defined_unary_funcs
    57-element Array{Symbol,1}:
     :sqrt
     :cbrt
     :abs2
     :inv
     :log
     :log10
     :log2
     :log1p
     :exp
     :exp2
     :expm1
     :sin
     :cos
     :tan
     ⋮

Some of these functions may have already been manually optimized. To see what functions have already been done, go to ``src/GradientNumber.jl``, scroll down to the ``Special Cases`` section, and look at the functions under ``Manually Optimized`` (further optimizations to these functions are always welcome, if you can come up with something clever).

2. Write out the function's first, second, and third derivatives
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Let's say that you wanted to manually optimize the ``sqrt`` function (this function's already been done, but is a good example).

Pretty naively, we can write out the function and its derivatives evaluated at some value :math:`a`:

.. math::

    y_0 &= \sqrt a \\
    y_1 &= \frac{\delta}{\delta a} \sqrt a = \frac{1}{2 \sqrt a} \\
    y_2 &= \frac{\delta^2}{\delta a^2} \sqrt a = - \frac{1}{4 a^{3/2}} \\
    y_3 &= \frac{\delta^3}{\delta a^3} \sqrt a = - \frac{3}{8 a^{5/2}}

3. Write and optimize a test function to compute the derivatives
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Let's write the function and derivatives from the previous section in Julia code:

.. code-block:: julia

    function sqrt_test(a)
        sqrt_a = sqrt(a)
        deriv1 = 1/(2*sqrt(a))
        deriv2 = 1/(4*sqrt(a)^3)
        deriv3 = -3/(8*sqrt(a)^5)
        return sqrt_a, deriv1, deriv2, deriv3
    end

Now, our goal is to minimize the number of operations used in ``sqrt_test``. For example, here's an optimized version of ``sqrt_test``:

.. code-block:: julia

    function sqrt_test(a)
        sqrt_a = sqrt(a)
        deriv1 = 0.5 / sqrt_a
        sqrt_a_cb = a * sqrt_a
        deriv2 = -0.25 / sqrt_a_cb
        deriv3 = 0.375 / (a * sqrt_a_cb)
        return sqrt_a, deriv1, deriv2, deriv3
    end

Note the variable reuse and precomputation of literal values.

3. Implement the function on the ``ForwardDiffNumber`` types
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Next, we want to implement the optimized operations from ``sqrt_test`` on all the ``ForwardDiffNumber`` types (it's actually pretty easy).

Let's start with ``GradientNumber``. Go to ``src/GradientNumber.jl``, scroll down to the ``Special Cases`` section, and look at the functions under ``Manually Optimized``. We're going to add this definition for the ``sqrt`` function (keep in mind that :math:`a \to` ``value(g)``):

.. code-block:: julia

    function sqrt(g::GradientNumber)
        sqrt_a = sqrt(value(g)) # sqrt_a from sqrt_test
        deriv = 0.5 / sqrt_a # deriv1 from sqrt_test
        return gradnum_from_deriv(g, sqrt_a, deriv)
    end

The body of the function is mostly just the value and first derivative calculations we've already figured out from writing ``sqrt_test``. The interesting line is the ``return`` statement; ``gradnum_from_deriv`` takes in the original ``GradientNumber`` (``g``), as its first argument, the new ``a`` value (``qrt_a``) as it's second, and the first derivative as its third. The resulting ``GradientNumber`` is then constructed from this information.


To define the optimized version of ``sqrt`` on ``HessianNumber`` and ``TensorNumber``, we basically do the same as the above, adding extra derivatives as necessary.

In ``src/HessianNumber.jl``, under the ``Special Cases``/``Manually Optimized`` section (:math:`a \to` ``value(h)``):

.. code-block:: julia

    function sqrt(h::HessianNumber)
        sqrt_a = sqrt(value(h))
        deriv1 = 0.5 / sqrt_a
        deriv2 = -0.25 / (a * sqrt_a)
        return hessnum_from_deriv(h, sqrt_a, deriv1, deriv2)
    end

In ``src/TensorNumber.jl``, under the ``Special Cases``/``Manually Optimized`` section (:math:`a \to` ``value(t)``):

.. code-block:: julia

    function sqrt(t::TensorNumber)
        sqrt_a = sqrt(value(t))
        deriv1 = 0.5 / sqrt_a
        sqrt_a_cb = a * sqrt_a
        deriv2 = -0.25 / sqrt_a_cb
        deriv3 = 0.375 / (a * sqrt_a_cb)
        return tensnum_from_deriv(t, sqrt_a, deriv1, deriv2, deriv3)
    end

4. Run Tests
++++++++++++

The functions in ``ForwardDiff.auto_defined_unary_funcs`` are automatically tested as part of ForwardDiff.jl's test suite, so you don't need to write tests yourself. Go ahead and test your changes by running ``Pkg.test("ForwardDiff")``.

If everything passes, you can submit a PR to the ForwardDiff.jl repository to share your work!

Implementing New Functions
--------------------------

Unary Functions Via Calculus.jl
+++++++++++++++++++++++++++++++

The easiest way to add support for a new function is actually to define a derivative rule for the function in Calculus.jl's `symbolic differentiation code`_, which ForwardDiff.jl then uses to generate the function's definition on the ``ForwardDiffNumber`` types. To accomplish this:

1. Open an issue in ForwardDiff.jl with the title "Supporting f(x)" (obviously replacing "f(x)"" with the function you wish to support).
2. Open a PR to Calculus.jl that adds the relevant differentiation rule(s) and tests. In the PR's description, be sure to mention the relevant ForwardDiff.jl issue such that GitHub links the two.
3. Once the PR to Calculus.jl is accepted, we can check to make sure that the function works appropriately in ForwardDiff.jl. If it does, then you're done, and the issue in ForwardDiff can be considered resolved!

.. _`symbolic differentiation code`: https://github.com/johnmyleswhite/Calculus.jl/blob/master/src/differentiate.jl#L115

Manually Adding Functions to ForwardDiff.jl
+++++++++++++++++++++++++++++++++++++++++++

The process for manually adding functions to ForwardDiff.jl without going through Calculus.jl is essentially the same as the process for manually optimizing existing functions (documented above). The only additional step is that you'll have to manually write tests for the function.

ForwardDiff.jl's `existing test suite`_ is full of examples demonstrating how to write tests for the package. You'll have to add tests for all subtypes of ``ForwardDiffNumber``. These tests should go under the corresponding files' "Special Cases" section.

.. _`existing test suite`: https://github.com/JuliaDiff/ForwardDiff.jl/tree/master/test
