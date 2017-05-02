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

Adding New Derivative Definitions
---------------------------------

In general, new derivative implementations for ``Dual`` are automatically defined via
simple symbolic rules. ForwardDiff accomplishes this by looping over the `the function names
listed in the RealInterface package`_, and for every function (and relevant arity), it
attempts to generate a ``Dual`` definition by applying the `symbolic rules provided by the
DiffBase package`_. Conveniently, these auto-generated definitions are also automatically
tested.

Thus, in order to add a new derivative implementation for ``Dual``, you should do the
following:

1. Make sure the name of the function is appropriately listed in the RealInterface package
2. Define the appropriate derivative rule(s) in DiffBase
3. Check that calling the function on ``Dual`` instances delivers the desired result.

Depending on the arity of your function and its category in RealInterface,
ForwardDiff's auto-definition mechanism might need to be expanded to include it.
If this is the case, ForwardDiff's maintainers can help you out.

.. _`the function names listed in the RealInterface package`: https://github.com/jrevels/RealInterface.jl
.. _`symbolic rules provided by the DiffBase package`: https://github.com/JuliaDiff/DiffBase.jl/blob/master/src/rules.jl
