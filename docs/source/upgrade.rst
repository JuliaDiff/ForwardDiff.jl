Upgrading from ForwardDiff v0.1 to v0.2
=======================================

A few API changes have occured between ForwardDiff v0.1 and v0.2. This document provides
some examples to help you transform old ForwardDiff code into new ForwardDiff code.

Setting Chunk Size
------------------

.. code-block:: julia

    # old way
    ForwardDiff.gradient(f, x; chunk_size = 10)

    # new way
    ForwardDiff.gradient(f, x, Chunk{10}())

Retrieving Lower-Order Results
------------------------------

For more detail, see our documentation on `retrieving lower-order results
<lower_order_results.html>`_.

.. code-block:: julia

    # old way
    answer, results = ForwardDiff.hessian(f, x, AllResults)
    v = ForwardDiff.value(results)
    g = ForwardDiff.gradient(results)
    h = ForwardDiff.hessian(results) # == answer

    # new way
    out = HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    v = ForwardDiff.value(out)
    g = ForwardDiff.gradient(out)
    h = ForwardDiff.hessian(out)

Creating Differentiation Functions
----------------------------------

ForwardDiff v0.2 no longer supports automatic generation of differentiation functions.
Instead, users explicitly define their own functions using ForwardDiff's API. This leads to
clearer code, less "magic", and more flexibility. To learn how about ForwardDiff's API
functions, see `our API documentation <api.html>`_.

.. code-block:: julia

    # old way - derivative function of f
    df = ForwardDiff.derivative(f)

    # new way
    df = x -> ForwardDiff.derivative(f, x)

.. code-block:: julia

    # old way - in-place gradient function of f
    gf! = ForwardDiff.gradient(f, mutates = true)

    # new way
    gf! = (out, x) -> ForwardDiff.gradient!(out, f, x)

.. code-block:: julia

    # old way - in-place Jacobian function of f!(y, x):
    jf! = ForwardDiff.jacobian(f!, mutates = true, output_length = length(y))

    # new way
    jf! = (out, y, x) -> ForwardDiff.jacobian!(out, f!, y, x)
