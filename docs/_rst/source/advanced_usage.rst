Advanced Usage Guide
====================

This document describes several techniques and features that can be used in conjunction
ForwardDiff's basic API in order to fine-tune calculations and increase performance.

Accessing Lower-Order Results
-----------------------------

Let's say you want to calculate the value, gradient, and Hessian of some function ``f`` at
an input ``x``. You could execute ``f(x)``, ``ForwardDiff.gradient(f, x)`` and
``ForwardDiff.hessian(f, x)``, but that would be a **horribly redundant way to  accomplish
this task!**

In the course of calculating higher-order derivatives, ForwardDiff ends up calculating all
the lower-order derivatives and primal value ``f(x)``. To retrieve these results in one fell
swoop, you can utilize the DiffResult API provided by the DiffBase package. To learn how to
use this functionality, please consult the `relevant documentation
<http://www.juliadiff.org/DiffBase.jl/diffresult/>`_.

Note that running ``using ForwardDiff`` will automatically bring the ``DiffBase`` module
into scope, and that all mutating ForwardDiff API methods support the DiffResult API.
In other words, API methods of the form ``ForwardDiff.method!(out, args...)`` will
work appropriately if ``out`` is a ``DiffResult``.

Configuring Chunk Size
----------------------

ForwardDiff performs partial derivative evaluation on one "chunk" of the input vector at a
time. Each differentation of a chunk requires a call to the target function as well as
additional memory proportional to the square of the chunk's size. Thus, a smaller chunk size
makes better use of memory bandwidth at the cost of more calls to the target function, while
a larger chunk size reduces calls to the target function at the cost of more memory
bandwidth.

For example:

.. code-block:: julia

    julia> import ForwardDiff

    # let's use a Rosenbrock function as our target function
    julia> function rosenbrock(x)
               a = one(eltype(x))
               b = 100 * a
               result = zero(eltype(x))
               for i in 1:length(x)-1
                   result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
               end
               return result
           end
    rosenbrock (generic function with 1 method)

    # input vector
    julia> x = rand(10000);

    # output buffer
    julia> out = similar(x);

    # construct Config with chunk size of 1
    julia> cfg1 = ForwardDiff.Config{1}(x);

    # construct Config with chunk size of 4
    julia> cfg4 = ForwardDiff.Config{4}(x);

    # construct Config with chunk size of 10
    julia> cfg10 = ForwardDiff.Config{10}(x);

    # (input length of 10000) / (chunk size of 1) = (10000 1-element chunks)
    julia> @time ForwardDiff.gradient!(out, rosenbrock, x, cfg1);
      0.408305 seconds (4 allocations: 160 bytes)

    # (input length of 10000) / (chunk size of 4) = (2500 4-element chunks)
    julia> @time ForwardDiff.gradient!(out, rosenbrock, x, cfg4);
      0.295764 seconds (4 allocations: 160 bytes)

    # (input length of 10000) / (chunk size of 10) = (1000 10-element chunks)
    julia> @time ForwardDiff.gradient!(out, rosenbrock, x, cfg10);
      0.267396 seconds (4 allocations: 160 bytes)

If you do not explicity provide a chunk size, ForwardDiff will try to guess one for you
based on your input vector:

.. code-block:: julia

    # The Config constructor will automatically select a
    # chunk size in one is not explicitly provided
    julia> cfg = ForwardDiff.Config(x);

    julia> @time ForwardDiff.gradient!(out, rosenbrock, x, cfg);
    0.266920 seconds (4 allocations: 160 bytes)

If your input dimension is a constant, you should explicitly select a chunk size rather than
relying on ForwardDiff's heuristic. There are two reasons for this. The first is that
ForwardDiff's heuristic depends only on the input dimension, whereas in reality the optimal
chunk size will also depend on the target function. The second is that ForwardDiff's
heuristic is inherently type-unstable, which can cause the entire call to be type-unstable.

If your input dimension is a runtime variable, you can rely on ForwardDiff's heuristic
without sacrificing type stability by manually asserting the output type, or - even better -
by using the in-place API functions:

.. code-block:: julia

    # will be type-stable since you're asserting the output type
    ForwardDiff.gradient(rosenbrock, x)::Vector{Float64}

    # will be type-stable since `out` is returned, and Julia knows the type of `out`
    ForwardDiff.gradient!(out, rosenbrock, x)

One final question remains: How should you select a chunk size? The answer is essentially
"perform your own benchmarks and see what works best for your use case." As stated before,
the optimal chunk size is heavily dependent on the target function and length of the input
vector.

When selecting a chunk size, keep in mind that the maximum allowed size is ``10`` (to
change this, you can alter the ``MAX_CHUNK_SIZE`` constant in ForwardDiff's source and
reload the package). Also, it is usually best to pick a chunk sizes which divides evenly
into the input dimension. Otherwise, ForwardDiff has to construct and utilize an extra
"remainder" chunk to complete the calculation.

Hessian of a vector-valued function
-----------------------------------

While ForwardDiff does not have a built-in function for taking Hessians of vector-valued
functions, you can easily compose calls to ``ForwardDiff.jacobian`` to accomplish this.
For example:

.. code-block:: julia

    julia> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(sin, x), [1,2,3])
    9×3 Array{Float64,2}:
     -0.841471   0.0        0.0
     -0.0       -0.0       -0.0
     -0.0       -0.0       -0.0
     0.0        0.0        0.0
     -0.0       -0.909297  -0.0
     -0.0       -0.0       -0.0
     0.0        0.0        0.0
     -0.0       -0.0       -0.0
     -0.0       -0.0       -0.14112

Since this functionality is composed from ForwardDiff's existing API rather than built into
it, you're free to construct a ``vector_hessian`` function which suits your needs. For
example, if you require the shape of the output to be a tensor rather than a block matrix,
you can do so with a ``reshape`` (note that ``reshape`` does not copy data, so it's not an
expensive operation):

.. code-block:: julia

    julia> function vector_hessian(f, x)
           n = length(x)
           out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
           return reshape(out, n, n, n)
       end
    vector_hessian (generic function with 1 method)

    julia> vector_hessian(sin, [1, 2, 3])
    3×3×3 Array{Float64,3}:
    [:, :, 1] =
     -0.841471   0.0   0.0
     -0.0       -0.0  -0.0
     -0.0       -0.0  -0.0

    [:, :, 2] =
      0.0   0.0        0.0
     -0.0  -0.909297  -0.0
     -0.0  -0.0       -0.0

    [:, :, 3] =
      0.0   0.0   0.0
     -0.0  -0.0  -0.0
     -0.0  -0.0  -0.14112

Likewise, you could write a version of ``vector_hessian`` which supports functions of the
form ``f!(y, x)``, or perhaps an in-place Jacobian with ``ForwardDiff.jacobian!``.

SIMD Vectorization
------------------

Many operations on ForwardDiff's dual numbers are amenable to `SIMD vectorization
<https://en.wikipedia.org/wiki/SIMD#Hardware>`_. For some ForwardDiff benchmarks, we've
seen SIMD vectorization yield `speedups of almost 3x
<https://github.com/JuliaDiff/ForwardDiff.jl/issues/98#issuecomment-253149761>`_.

To enable SIMD optimizations, start your Julia process with the ``-O3`` flag. This flag
enables `LLVM's SLPVectorizerPass
<http://llvm.org/docs/Vectorizers.html#the-slp-vectorizer>`_ during compilation, which
attempts to automatically insert SIMD instructions where possible for certain arithmetic
operations.

Here's an example of LLVM bitcode generated for an addition of two ``Dual`` numbers without
SIMD instructions (i.e. not starting Julia with ``-O3``):

.. code-block:: julia

    julia> using ForwardDiff: Dual

    julia> a = Dual(1., 2., 3., 4.)
    Dual(1.0,2.0,3.0,4.0)

    julia> b = Dual(5., 6., 7., 8.)
    Dual(5.0,6.0,7.0,8.0)

    julia> @code_llvm a + b

    define void @"julia_+_70852"(%Dual* noalias sret, %Dual*, %Dual*) #0 {
    top:
      %3 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 0
      %4 = load double, double* %3, align 8
      %5 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 0
      %6 = load double, double* %5, align 8
      %7 = fadd double %4, %6
      %8 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 1
      %9 = load double, double* %8, align 8
      %10 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 1
      %11 = load double, double* %10, align 8
      %12 = fadd double %9, %11
      %13 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 2
      %14 = load double, double* %13, align 8
      %15 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 2
      %16 = load double, double* %15, align 8
      %17 = fadd double %14, %16
      %18 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 0
      %19 = load double, double* %18, align 8
      %20 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 0
      %21 = load double, double* %20, align 8
      %22 = fadd double %19, %21
      %23 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 0
      store double %22, double* %23, align 8
      %24 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 0
      store double %7, double* %24, align 8
      %25 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 1
      store double %12, double* %25, align 8
      %26 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 2
      store double %17, double* %26, align 8
      ret void
    }

If we start up Julia with ``-O3`` instead, the call to ``@code_llvm`` will show that LLVM
can SIMD-vectorize the addition:

.. code-block:: julia

    julia> @code_llvm a + b

    define void @"julia_+_70842"(%Dual* noalias sret, %Dual*, %Dual*) #0 {
    top:
      %3 = bitcast %Dual* %1 to <4 x double>*            # cast the Dual to a SIMD-able LLVM vector
      %4 = load <4 x double>, <4 x double>* %3, align 8
      %5 = bitcast %Dual* %2 to <4 x double>*
      %6 = load <4 x double>, <4 x double>* %5, align 8
      %7 = fadd <4 x double> %4, %6                      # SIMD add
      %8 = bitcast %Dual* %0 to <4 x double>*
      store <4 x double> %7, <4 x double>* %8, align 8
      ret void
    }

Note that whether or not SIMD instructions can actually be used will depend on your machine
and Julia build. For example, pre-built Julia binaries might not emit vectorized LLVM
bitcode. To overcome this specific issue, you can `locally rebuild Julia's system image
<http://docs.julialang.org/en/latest/devdocs/sysimg/>`_.
