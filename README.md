Note: This README describes the current version of the package on the `master` branch, which is only supported on Julia v0.4. The documentation for the old version of this package can still be found [here](http://forwarddiffjl.readthedocs.org/en/latest/).

[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl) [![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=master)

# ForwardDiff.jl

The `ForwardDiff` package provides a type-based implementation of forward mode automatic differentiation (FAD) in Julia. [The wikipedia page on automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of FAD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

## What can I do with this package?

This package contains methods to take derivatives, gradients, Jacobians, and Hessians of native Julia functions (or any callable object, really). While performance varies depending on the functions you evaluate, this package generally outperforms non-AD methods in memory usage, speed, and accuracy.

A third-order generalization of the Hessian is also implemented (see the `tensor` method). 

For now, we only support for functions involving `T<:Real`s, but we believe extension to numbers of type `T<:Complex` is possible.

## Usage Example

```julia
julia> using ForwardDiff

julia> f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);

julia> x = rand(5) # small size for example's sake
5-element Array{Float64,1}:
 0.986403
 0.140913
 0.294963
 0.837125
 0.650451

julia> g = ForwardDiff.gradient(f); # g = ∇f

julia> g(x)
5-element Array{Float64,1}:
 1.01358
 2.50014
 1.72574
 1.10139
 1.2445

julia> j = ForwardDiff.jacobian(g); # j = J(∇f)

julia> j(x)
5x5 Array{Float64,2}:
 0.585111  3.48083  1.7706    0.994057  1.03257
 3.48083   1.06079  5.79299   3.25245   3.37871
 1.7706    5.79299  0.423981  1.65416   1.71818
 0.994057  3.25245  1.65416   0.251396  0.964566
 1.03257   3.37871  1.71818   0.964566  0.140689

julia> ForwardDiff.hessian(f, x) # H(f)(x) == J(∇f)(x), as expected 
5x5 Array{Float64,2}:
 0.585111  3.48083  1.7706    0.994057  1.03257
 3.48083   1.06079  5.79299   3.25245   3.37871
 1.7706    5.79299  0.423981  1.65416   1.71818
 0.994057  3.25245  1.65416   0.251396  0.964566
 1.03257   3.37871  1.71818   0.964566  0.140689
 ```

## API

#### Derivative of `f(x::Number) → Number` or `f(x::Number) → Array`

- **`derivative!(output::Array, f, x::Number)`**
    
    Compute `f'(x)`, storing the output in `output`.

- **`derivative(f, x::Number)`**
    
    Compute `f'(x)`.

- **`derivative(f; mutates=false)`**
    
    Return the function `f'`. If `mutates=false`, then the returned function has the form `derivf(x) -> derivative(f, x)`. If `mutates = true`, then the returned function has the form `derivf!(output, x) -> derivative!(output, f, x)`.

#### Gradient of `f(x::Vector) → Number`

- **`gradient!(output::Vector, f, x::Vector)`**

    Compute `∇f(x)`, storing the output in `output`.

- **`ForwardDiff.gradient{T}(f, x::Vector{T})`**

    Compute `∇f(x)`, where `T` is the element type of both the input and output. `ForwardDiff` must be used as a qualifier when calling `gradient` to avoid conflict with `Base.gradient`.

- **`ForwardDiff.gradient(f; mutates=false)`**

    Return the function `∇f`. If `mutates=false`, then the returned function has the form `gradf(x) -> gradient(f, x)`. If `mutates = true`, then the returned function has the form `gradf!(output, x) -> gradient!(output, f, x)`. By default, `mutates` is set to `false`. `ForwardDiff` must be used as a qualifier when calling `gradient` to avoid conflict with `Base.gradient`.

#### Jacobian of `f(x:Vector) → Vector`

- **`jacobian!(output::Matrix, f, x::Vector)`**

    Compute `J(f(x))`, storing the output in `output`.

- **`jacobian{T}(f, x::Vector{T})`**

    Compute `J(f(x))`, where `T` is the element type of both the input and output.

- **`jacobian(f; mutates=false)`**

    Return the function `J(f)`. If `mutates=false`, then the returned function has the form `jacf(x) -> jacobian(f, x)`. If `mutates = true`, then the returned function has the form `jacf!(output, x) -> jacobian!(output, f, x)`. By default, `mutates` is set to `false`.

#### Hessian of `f(x::Vector) → Number`

- **`hessian!(output::Matrix, f, x::Vector)`**

    Compute `H(f(x))`, storing the output in `output`.

- **`hessian{T}(f, x::Vector{T})`**

    Compute `H(f(x))`, where `T` is the element type of both the input and output.

- **`hessian(f; mutates=false)`**

    Return the function `H(f)`. If `mutates=false`, then the returned function has the form `hessf(x) -> hessian(f, x, S)`. If `mutates = true`, then the returned function has the form `hessf!(output, x) -> hessian!(output, f, x)`. By default, `mutates` is set to `false`.

#### Third-order Taylor series term of `f(x::Vector) → Number`

[This Math StackExchange post](http://math.stackexchange.com/questions/556951/third-order-term-in-taylor-series) actually has an answer that explains this term fairly clearly.

- **`tensor!{T}(output::Array{T,3}, f, x::Vector)`**

    Compute `∑D³f(x)`, storing the output in `output`.

- **`tensor{T}(f, x::Vector{T})`**

    Compute `∑D³f(x)`, where `T` is the element type of both the input and output.

- **`tensor(f; mutates=false)`**

    Return the function ``∑D³f``. If `mutates=false`, then the returned function has the form `tensf(x) -> tensor(f, x)`. If `mutates = true`, then the returned function has the form `tensf!(output, x) -> tensor!(output, f, x)`. By default, `mutates` is set to `false`.

## Performance Tips

#### Type stability

Make sure that your target function is [type-stable](http://julia.readthedocs.org/en/latest/manual/performance-tips/#write-type-stable-functions). Type instability in the target function can cause slowness, and in some cases, errors. If you get an error that looks like this:

```julia
ERROR: TypeError: typeassert: ...
```

It might be because Julia's type inference can't predict the output of your target function to be the output expected by ForwardDiff (these expectations are outlined in the API above).

#### Caching Options

If you're going to be repeatedly evaluating the gradient/Hessian/etc. of a function, it may be worth it to generate the corresponding method beforehand rather than call `hessian(f, x)` a bunch of times. For example, this:
    
```julia
inputs = [rand(1000) for i in 1:100]
for x in inputs
    hess = hessian(f, x)
    ... # do something with hess
end
```

should really be written like this:

```julia
h = hessian(f) # generate H(f) first
inputs = [rand(1000) for i in 1:100]
for x in inputs
    hess = h(x)
    ... # do something with hess
end
```

The reason this is the case is that `hessian(f, x)` (and `hessian!(output, f, x)`, as well as the other methods like `gradient`/`jacobian`/etc.) must create various temporary "work arrays" over the course of evaluation. Generating `h = hessian(f)` *first* allows the temporary arrays to be cached in subsequent calls to `h`, saving both time and memory over the course of the loop.

This caching can also be handled manually, if one wishes, by utilizing the provided `ForwardDiffCache` type and the `cache` keyword argument:

```julia
my_cache = ForwardDiffCache() # make new cache to pass in to our function
inputs = [rand(1000) for i in 1:100]
for x in inputs
    # just as efficient as pre-generating h because it can reuse my_cache
    hess = hessian(f, x, cache=my_cache) 
    ... # do something with hess
end
```

The `cache` keyword argument is supported for all ForwardDiff methods which take in a target function and input.

Manually passing in a `FowardDiffCache` is useful when `f` might change over the context of the calculation, in which case you wouldn't want to have to keep re-generating `hessian(f)` over and over.

#### Chunk-based calculation modes

If the dimensions you're working in are very large, ForwardDiff supports calculation of gradients/Jacobians/etc. by "chunks" of the input vector rather than performing the entire computation at once. This mode is triggered by use of the `chunk_size` keyword argument, like so:

```julia
x = rand(1000000)
jacobian(f, x, chunk_size=10) # perform calculation only 10 elements at a time to save memory
```

This also applies to pre-computed FAD methods, and mutating methods:

```julia
j! = jacobian(f, mutates=true)
j!(output, x, chunk_size=10)
```

The given `chunk_size` must always evenly divide the length of the input vector:

```julia
julia> hessian(f, rand(100), chunk_size=11)
ERROR: AssertionError: Length of input vector is indivisible by chunk size (length(x) = 100, chunk size = 11)
 in check_chunk_size at /Users/jarrettrevels/.julia/ForwardDiff/src/fad_api/fad_api.jl:19
 in _calc_hessian! at /Users/jarrettrevels/.julia/ForwardDiff/src/fad_api/hessian.jl:48
```

Thus, chunking of input vectors whose length is a prime number is unsupported. We're currently working on removing this limitation.

Note that it is generally always much faster to **not** provide a `chunk_size`. This option is provided for the cases in which performing an entire calculation at once would consume too much memory.
