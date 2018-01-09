# Advanced Usage Guide

This document describes several techniques and features that can be used in conjunction
ForwardDiff's basic API in order to fine-tune calculations and increase performance.

## Retrieving Lower-Order Results

Let's say you want to calculate the value, gradient, and Hessian of some function `f` at
an input `x`. You could execute `f(x)`, `ForwardDiff.gradient(f, x)` and
`ForwardDiff.hessian(f, x)`, but that would be a **horribly redundant way to accomplish
this task!**

In the course of calculating higher-order derivatives, ForwardDiff ends up calculating all
the lower-order derivatives and primal value `f(x)`. To retrieve these results in one fell
swoop, you can utilize the [DiffResults](https://github.com/JuliaDiff/DiffResults.jl) API.

All mutating ForwardDiff API methods support the DiffResults API. In other words, API
methods of the form `ForwardDiff.method!(out, args...)` will work appropriately if
`isa(out, DiffResults.DiffResult)`.

## Configuring Chunk Size

ForwardDiff performs partial derivative evaluation on one "chunk" of the input vector at a
time. Each differentiation of a chunk requires a call to the target function as well as
additional memory proportional to the square of the chunk's size. Thus, a smaller chunk size
makes better use of memory bandwidth at the cost of more calls to the target function, while
a larger chunk size reduces calls to the target function at the cost of more memory
bandwidth.

For example:

```julia
julia> using ForwardDiff: GradientConfig, Chunk, gradient!

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

# construct GradientConfig with chunk size of 1
julia> cfg1 = GradientConfig(rosenbrock, x, Chunk{1}());

# construct GradientConfig with chunk size of 4
julia> cfg4 = GradientConfig(rosenbrock, x, Chunk{4}());

# construct GradientConfig with chunk size of 10
julia> cfg10 = GradientConfig(rosenbrock, x, Chunk{10}());

# (input length of 10000) / (chunk size of 1) = (10000 1-element chunks)
julia> @time gradient!(out, rosenbrock, x, cfg1);
  0.775139 seconds (4 allocations: 160 bytes)

# (input length of 10000) / (chunk size of 4) = (2500 4-element chunks)
julia> @time gradient!(out, rosenbrock, x, cfg4);
  0.386459 seconds (4 allocations: 160 bytes)

# (input length of 10000) / (chunk size of 10) = (1000 10-element chunks)
julia> @time gradient!(out, rosenbrock, x, cfg10);
  0.282529 seconds (4 allocations: 160 bytes)
```

If you do not explicity provide a chunk size, ForwardDiff will try to guess one for you
based on your input vector:

```julia
# The GradientConfig constructor will automatically select a
# chunk size in one is not explicitly provided
julia> cfg = ForwardDiff.GradientConfig(rosenbrock, x);

julia> @time ForwardDiff.gradient!(out, rosenbrock, x, cfg);
  0.281853 seconds (4 allocations: 160 bytes)
```

If your input dimension is constant across calls, you should explicitly select a chunk size
rather than relying on ForwardDiff's heuristic. There are two reasons for this. The first is
that ForwardDiff's heuristic depends only on the input dimension, whereas in reality the
optimal chunk size will also depend on the target function. The second is that ForwardDiff's
heuristic is inherently type-unstable, which can cause the entire call to be type-unstable.

If your input dimension is a runtime variable, you can rely on ForwardDiff's heuristic
without sacrificing type stability by manually asserting the output type, or - even better -
by using the in-place API functions:

```julia
# will be type-stable since you're asserting the output type
ForwardDiff.gradient(rosenbrock, x)::Vector{Float64}

# will be type-stable since `out` is returned, and Julia knows the type of `out`
ForwardDiff.gradient!(out, rosenbrock, x)
```

One final question remains: How should you select a chunk size? The answer is essentially
"perform your own benchmarks and see what works best for your use case." As stated before,
the optimal chunk size is heavily dependent on the target function and length of the input
vector.

Note that it is usually best to pick a chunk size which divides evenly into the input
dimension. Otherwise, ForwardDiff has to construct and utilize an extra "remainder" chunk to
complete the calculation.

## Fixing NaN/Inf Issues

ForwardDiff's default behavior is to return `NaN` for undefined derivatives (or otherwise
mirror the behavior of the function in `Base`, if it would return an error). This is
usually the correct thing to do, but in some cases can erroneously "poison" values which
aren't sensitive to the input and thus cause ForwardDiff to incorrectly return `NaN` or
`Inf` derivatives. For example:

```julia
# the dual number's perturbation component is zero, so this
# variable should not propagate derivative information
julia> log(ForwardDiff.Dual{:tag}(0.0, 0.0))
Dual{:tag}(-Inf,NaN) # oops, this NaN should be 0.0
```

Here, ForwardDiff computes the derivative of `log(0.0)` as `NaN` and then propagates
this derivative by multiplying it by the perturbation component. Usually, ForwardDiff can
rely on the identity `x * 0.0 == 0.0` to prevent the derivatives from propagating when
the perturbation component is `0.0`. However, this identity doesn't hold if `isnan(y)`
or `isinf(y)`, in which case a `NaN` derivative will be propagated instead.

It is possible to fix this behavior by checking that the perturbation component is zero
before attempting to propagate derivative information, but this check can noticeably
decrease performance (~5%-10% on our benchmarks).

In order to preserve performance in the majority of use cases, ForwardDiff disables this
check by default. If your code is affected by this `NaN` behvaior, you can enable
ForwardDiff's `NaN`-safe mode by setting the `NANSAFE_MODE_ENABLED` constant to `true` in
ForwardDiff's source.

In the future, we plan on allowing users and downstream library authors to dynamically
enable [NaN`-safe mode via the `AbstractConfig`
API](https://github.com/JuliaDiff/ForwardDiff.jl/issues/181).

## Hessian of a vector-valued function

While ForwardDiff does not have a built-in function for taking Hessians of vector-valued
functions, you can easily compose calls to `ForwardDiff.jacobian` to accomplish this.
For example:

```julia
julia> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(cumprod, x), [1,2,3])
9×3 Array{Int64,2}:
 0  0  0
 0  1  0
 0  3  2
 0  0  0
 1  0  0
 3  0  1
 0  0  0
 0  0  0
 2  1  0
```

Since this functionality is composed from ForwardDiff's existing API rather than built into
it, you're free to construct a `vector_hessian` function which suits your needs. For
example, if you require the shape of the output to be a tensor rather than a block matrix,
you can do so with a `reshape` (note that `reshape` does not copy data, so it's not an
expensive operation):

```julia
julia> function vector_hessian(f, x)
       n = length(x)
       out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
       return reshape(out, n, n, n)
   end
vector_hessian (generic function with 1 method)

julia> vector_hessian(cumprod, [1, 2, 3])
3×3×3 Array{Int64,3}:
[:, :, 1] =
 0  0  0
 0  1  0
 0  3  2

[:, :, 2] =
 0  0  0
 1  0  0
 3  0  1

[:, :, 3] =
 0  0  0
 0  0  0
 2  1  0
```

Likewise, you could write a version of `vector_hessian` which supports functions of the
form `f!(y, x)`, or perhaps an in-place Jacobian with `ForwardDiff.jacobian!`.

## SIMD Vectorization

Many operations on ForwardDiff's dual numbers are amenable to [SIMD
vectorization](https://en.wikipedia.org/wiki/SIMD#Hardware). For some ForwardDiff
benchmarks, we've seen SIMD vectorization yield [speedups of almost
3x](https://github.com/JuliaDiff/ForwardDiff.jl/issues/98#issuecomment-253149761).

To enable SIMD optimizations, start your Julia process with the `-O3` flag. This flag
enables [LLVM's SLPVectorizerPass](http://llvm.org/docs/Vectorizers.html#the-slp-vectorizer)
during compilation, which attempts to automatically insert SIMD instructions where possible
for certain arithmetic operations.

Here's an example of LLVM bitcode generated for an addition of two `Dual` numbers without
SIMD instructions (i.e. not starting Julia with `-O3`):

```julia
julia> using ForwardDiff: Dual

julia> a = Dual(1., 2., 3., 4.)
Dual{Nothing}(1.0,2.0,3.0,4.0)

julia> b = Dual(5., 6., 7., 8.)
Dual{Nothing}(5.0,6.0,7.0,8.0)

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
```

If we start up Julia with `-O3` instead, the call to `@code_llvm` will show that LLVM
can SIMD-vectorize the addition:

```julia
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
```

Note that whether or not SIMD instructions can actually be used will depend on your machine
and Julia build. For example, pre-built Julia binaries might not emit vectorized LLVM
bitcode. To overcome this specific issue, you can [locally rebuild Julia's system
image](http://docs.julialang.org/en/latest/devdocs/sysimg).

## Custom tags and tag checking

The `Dual` type includes a "tag" parameter indicating the particular function call to
which it belongs. This is to avoid a problem known as [_perturbation
confusion_](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) which can arise when
there are nested differentiation calls. Tags are automatically generated as part of the
appropriate config object, and the tag is checked when the config is used as part of a
differentiation call (`derivative`, `gradient`, etc.): an `InvalidTagException` will be
thrown if the incorrect config object is used.

This checking can sometimes be inconvenient, and there are certain cases where you may
want to disable this checking.

!!! warning
    Disabling tag checking should only be done with caution, especially if the code itself
    could be used inside another differentiation call.

1. (preferred) Provide an extra `Val{false}()` argument to the differentiation function, e.g.
   ```julia
   cfg = ForwarDiff.GradientConfig(g, x)
   ForwarDiff.gradient(f, x, cfg, Val{false}())
   ```
   If using as part of a library, the tag can be checked manually via
   ```julia
   ForwardDiff.checktag(cfg, g, x)
   ```

2. (discouraged) Construct the config object with `nothing` instead of a function argument, e.g.
   ```julia
   cfg = GradientConfig(nothing, x)
   gradient(f, x, cfg)
   ```