Chunk-mode vs. Vector-mode
==========================

What are these modes?
---------------------

By default, ForwardDiff.jl operates over the entire input vector simultaneously when evaluating partial derivatives. This mode, called **vector-mode**, minimizes the number of calls to the target function :math:`f`, but does so at a potentially high memory cost. As such, vector-mode is quite suitable for *small* input vectors, but *large* input vectors can incur heavy GC overhead.

To get around this problem, ForwardDiff.jl provides **chunk-mode**. In chunk-mode, partial derivative evaluation is performed on one "chunk" of the input vector at a time. This results in more calls to :math:`f`, but requires less memory and thus incurs less GC overhead.

How do I use these modes?
-------------------------

The user doesn't have to do anything to use vector-mode, since it is used by default.

To use chunk-mode, simply pass in a ``chunk_size`` keyword argument to any of ForwardDiff.jl's differentiation methods (except ``tensor``/``tensor!``, for which chunk-mode is not yet supported):

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

    julia> g = ForwardDiff.gradient(rosenbrock);

    julia> x = rand(16000);

    julia> @time g(x); # in reality, this is a post-warmup result
    2.895158 seconds (655.99 k allocations: 15.274 GB, 17.45% gc time)

    # g_chunk will only calculate 5 elements at a time to save memory
    julia> g_chunk = ForwardDiff.gradient(rosenbrock, chunk_size=5);

    julia> @time g_chunk(x); # also post-warmup
    0.651275 seconds (6.42 k allocations: 525.641 KB)

Note that the provided ``chunk_size`` must always evenly divide the length of the input vector:

.. code-block:: julia

    julia> ForwardDiff.hessian(rosenbrock, rand(100), chunk_size=11)
    ERROR: AssertionError: Length of input vector is indivisible by chunk size (length(x) = 100, chunk size = 11)

We're currently working on removing this limitation so that input vectors with prime lengths won't be at a disadvantage.

What mode/``chunk_size`` should I use?
--------------------------------------

The best mode/``chunk_size`` for any given problem is heavily dependent on the target function and length of the input vector. As such, one should generally perform their own benchmarks to determine whether to use vector-mode or chunk-mode (or to pick the appropriate ``chunk_size``).

Here are some tips:

- If your input vector is large (over ~1000 elements), consider using chunk-mode.

- If you find that a large portion of calculation time can be attributed to GC, consider using chunk-mode.

- The sweet spot for ``chunk_size`` will usually be between 1 and ~10, because calculations on large chunks can "clog" the stack. This behavior is due to ForwardDiff.jl's reliance on stack-allocated tuples for chunk-mode.
