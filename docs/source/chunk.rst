Configuring Partial Derivative Chunks
=====================================

ForwardDiff performs partial derivative evaluation on one "chunk" of the input vector at a
time. Each differentation of a chunk requires a call to the target function as well as
additional memory proportional to the square of the chunk's size. Thus, a smaller chunk size
makes better use of memory bandwidth at the cost of more calls to the target function, while
a larger chunk size reduces calls to the target function at the cost of more memory
bandwidth.

The user can specify the chunk size they wish to use by passing in ``Chunk{N}()`` as an
argument to the API functions, where ``N`` is the desired chunk size. For example:

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

    julia> x = rand(10000);

    # (input length of 10000) / (chunk size of 1) = (10000 1-element chunks)
    julia> @time ForwardDiff.gradient(rosenbrock, x, Chunk{1}());
      0.403879 seconds (11 allocations: 78.469 KB)

    # (input length of 10000) / (chunk size of 4) = (2500 4-element chunks)
    julia> @time ForwardDiff.gradient(rosenbrock, x, Chunk{4}());
      0.314482 seconds (11 allocations: 78.469 KB)

    # (input length of 10000) / (chunk size of 10) = (1000 10-element chunks)
    julia> @time ForwardDiff.gradient(rosenbrock, x, Chunk{10}());
      0.265994 seconds (11 allocations: 78.469 KB)

    # (input length of 10000) / (chunk size of 16) = (625 16-element chunks)
    julia> @time ForwardDiff.gradient(rosenbrock, x, Chunk{16}());
      0.294078 seconds (11 allocations: 78.469 KB)

If you do not explicity provide a chunk size, ForwardDiff will try to guess one for you
based on your input vector:

.. code-block:: julia

    julia> @time ForwardDiff.gradient(rosenbrock, x);
    0.265604 seconds (11 allocations: 78.469 KB)

If your input dimension is a constant, you should explicitly select a chunk size rather than
relying on ForwardDiff's heuristic. There are two reasons for this. The first is that
ForwardDiff's heuristic depends only on the input dimension, whereas in reality the optimal
chunk size will also depend on the target function. The second is that ForwardDiff's
heuristic is inherently type-unstable, which can cause the entire call to be type-unstable:

.. code-block:: julia

    # type-unstable if you don't manually provide a chunk size
    julia> @code_warntype ForwardDiff.gradient(rosenbrock, x);
    Variables:
      #self#::ForwardDiff.#gradient
      f::#rosenbrock
      x::Array{Float64,1}

    Body:
      begin
          return (#self#::ForwardDiff.#gradient)(f::#rosenbrock,x::Array{Float64,1},((Core.apply_type)(ForwardDiff.Chunk,(ForwardDiff.pickchunksize)(x::Array{Float64,1})::Int64)::Type{_<:ForwardDiff.Chunk})()::ForwardDiff.Chunk{N})::Any
      end::Any

    # type-stable if you manually provide a chunk size
    julia> @code_warntype ForwardDiff.gradient(rosenbrock, x, Chunk{10}());
    Variables:
      #self#::ForwardDiff.#gradient
      f::#rosenbrock
      x::Array{Float64,1}
      chunk::ForwardDiff.Chunk{10}

    Body:
      begin
          return (ForwardDiff.#gradient#21)(false,#self#::ForwardDiff.#gradient,f::#rosenbrock,x::Array{Float64,1},chunk::ForwardDiff.Chunk{10})::Array{Float64,1}
      end::Array{Float64,1}

If your input dimension is a runtime variable, you can rely on ForwardDiff's heuristic
without sacrificing type stability by manually asserting the output type, or - even better -
by using the in-place API functions:

.. code-block:: julia

    # will be type-stable since you're asserting the output type
    ForwardDiff.gradient(rosenbrock, x)::Vector{Float64}

    # will be type-stable since `out` is returned, and Julia knows the type of `out`
    ForwardDiff.gradient!(out, rosenbrock, x)

What chunk size should I use?
-----------------------------

The chunk size for any given problem is heavily dependent on the target function and length
of the input vector. As such, one should generally perform their own benchmarks to determine
which chunk size to use. Here are some tips:

- The max chunk size is ``20``, but usually the chunk size is most optimal at less than or equal to ``10``.

- The chunk size should generally divide evenly into the input dimension. Otherwise, ForwardDiff has to construct and utilize an extra "remainder" chunk to complete the calculation.
