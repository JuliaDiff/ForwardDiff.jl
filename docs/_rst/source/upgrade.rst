Upgrading from Older Versions of ForwardDiff
============================================

A few API changes have occured between ForwardDiff v0.1, v0.2, and v0.3 (the current
version). This document provides some examples to help you transform old ForwardDiff code
into new ForwardDiff code.

Unexported API Functions
------------------------

In order to avoid namespace conflicts with other packages, `ForwardDiff's API functions
<basic_api.html>`_ are no longer exported by default. Thus, you must now fully qualify the
functions to reference them:

.. code-block:: julia

    # old v0.1 style
    using ForwardDiff
    hessian(f, x)

    # current v0.3 style (since v0.2)
    using ForwardDiff
    ForwardDiff.hessian(f, x)

Setting Chunk Size
------------------

.. code-block:: julia

    # old v0.1 style
    ForwardDiff.gradient(f, x; chunk_size = 10)

    # old v0.2 style
    ForwardDiff.gradient(f, x, Chunk{10}())

    # current v0.3 style
    ForwardDiff.gradient(f, x, ForwardDiff.Options{10}(x))

Enabling Multithreading
-----------------------

.. code-block:: julia

    # old v0.1/v0.2 style
    ForwardDiff.gradient(f, x; multithread = true)

    # current v0.3 style
    ForwardDiff.gradient(f, x, ForwardDiff.Multithread(ForwardDiff.Options(x)))

Retrieving Lower-Order Results
------------------------------

For more detail, see our documentation on `retrieving lower-order results
<advanced_usage.html#Accessing_Lower_Order_Results>`_.

.. code-block:: julia

    # old v0.1 style
    answer, results = ForwardDiff.hessian(f, x, AllResults)
    v = ForwardDiff.value(results)
    g = ForwardDiff.gradient(results)
    h = ForwardDiff.hessian(results) # == answer

    # old v0.2 style
    out = HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    v = ForwardDiff.value(out)
    g = ForwardDiff.gradient(out)
    h = ForwardDiff.hessian(out)

    # current v0.3 style
    using DiffBase
    out = DiffBase.HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    v = DiffBase.value(out)
    g = DiffBase.gradient(out)
    h = DiffBase.hessian(out)

Higher-Order Differentiation
----------------------------

In order to maintain feature parity between all API functions, ForwardDiff no longer
provides the ``tensor`` function. Instead, users can take higher-order/higher-dimensional
derivatives by composing existing API functions. For example, here's how to reimplement
``tensor``:

.. code-block:: julia

    # old v0.1 style
    ForwardDiff.tensor(f, x)

    # current v0.3 style (since v0.2)
    function tensor(f, x)
        n = length(x)
        out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
        return reshape(out, n, n, n)
    end

    tensor(f, x)

Creating Differentiation Functions
----------------------------------

Since v0.2, ForwardDiff no longer supports automatic generation of differentiation
functions. Instead, users explicitly define their own functions using ForwardDiff's API.
This leads to clearer code, less "magic", and more flexibility. To learn how about
ForwardDiff's API functions, see `our API documentation <basic_api.html>`_.

.. code-block:: julia

    # old v0.1 style
    df = ForwardDiff.derivative(f)

    # current v0.3 style (since v0.2)
    df = x -> ForwardDiff.derivative(f, x)

.. code-block:: julia

    # old v0.1 style
    # in-place gradient function of f
    gf! = ForwardDiff.gradient(f, mutates = true)

    # current v0.3 style (since v0.2)
    gf! = (out, x) -> ForwardDiff.gradient!(out, f, x)

.. code-block:: julia

    # old v0.1 style
    # in-place Jacobian function of f!(y, x):
    jf! = ForwardDiff.jacobian(f!, mutates = true, output_length = length(y))

    # current v0.3 style (since v0.2)
    jf! = (out, y, x) -> ForwardDiff.jacobian!(out, f!, y, x)
