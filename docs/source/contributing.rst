How to Contribute
=================

There are a few fairly easy ways for newcomers to substantially improve ForwardDiff, and
they all revolve around **writing functions for** ``Dual`` **numbers**. This section
provides brief tutorials on how to make these contributions.

If you're new GitHub, here's an outline of the workflow you should use:

1. Fork ForwardDiff
2. Make a new branch on your fork, named after whatever changes you'll be making
3. Apply your code changes to the branch on your fork
4. When you're done, submit a PR to ForwardDiff to merge your fork into ForwardDiff's master branch.

Manually Optimizing Unary Functions
-----------------------------------

To see a list of functions to pick from, look at ``ForwardDiff.AUTO_DEFINED_UNARY_FUNCS``:

.. code-block:: julia

    julia> import ForwardDiff

    julia> ForwardDiff.AUTO_DEFINED_UNARY_FUNCS
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

Some of these functions may have already been manually optimized. To see what functions have
already been done, go to ``src/dual.jl``, scroll down to the ``Special Cases`` section, and
look at the functions under ``Manually Optimized`` (further optimizations to these functions
are always welcome, if you can come up with something clever).

The functions in ``ForwardDiff.AUTO_DEFINED_UNARY_FUNCS`` are automatically tested as part
of ForwardDiff's test suite, so you don't need to write tests yourself. You can test your
changes by running ``Pkg.test("ForwardDiff")``.

If everything passes, you can submit a PR to the ForwardDiff repository to share your work!

Implementing New Functions
--------------------------

Unary Functions Via Calculus.jl
+++++++++++++++++++++++++++++++

The easiest way to add support for a new function is actually to define a derivative rule
for the function in Calculus's `symbolic differentiation code`_, which ForwardDiff then uses
to generate the function's definition on ``Dual`` numbers. To accomplish this:

1. Open an issue in ForwardDiff with the title "Supporting f(x)" (obviously replacing "f(x)"" with the function you wish to support).
2. Open a PR to Calculus that adds the relevant differentiation rule(s) and tests. In the PR's description, be sure to mention the relevant ForwardDiff issue such that GitHub links the two.
3. Once the PR to Calculus is accepted, we can check to make sure that the function works appropriately in ForwardDiff. If it does, then you're done, and the issue in ForwardDiff can be considered resolved!

.. _`symbolic differentiation code`: https://github.com/johnmyleswhite/Calculus.jl/blob/master/src/differentiate.jl#L115

Manually Adding Functions to ForwardDiff
++++++++++++++++++++++++++++++++++++++++

Some functions aren't suitable for auto-definition via the Calculus package. An example of
such a function is the non-unary function ``atan2``, which is defined manually in
``ForwardDiff/src/dual.jl``.

The process for manually adding functions to ForwardDiff without going through Calculus.jl
is essentially the same as the process for manually optimizing existing functions, with the
additional requirement that you'll have to write the tests yourself. New tests for ``Dual``
numbers can be placed in `DualTest.jl`_.

.. _`DualTest.jl`: https://github.com/JuliaDiff/ForwardDiff.jl/tree/master/test/DualTest.jl
