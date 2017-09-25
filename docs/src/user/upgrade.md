# Upgrading from Older Versions

Each new minor release of ForwardDiff may introduce changes in the API or behavior. Here,
we'll provide some examples that highlight the key differences to help you upgrade from
older versions of ForwardDiff.

# Unexported API Functions

In order to avoid namespace conflicts with other packages, ForwardDiff's  [Differentiation
API](@ref) is no longer exported by default. Thus, you must now fully qualify the functions
to  reference them:

```julia
# ForwardDiff v0.1
using ForwardDiff
hessian(f, x)

# ForwardDiff v0.2 & above
using ForwardDiff
ForwardDiff.hessian(f, x)
```

# Using The `AbstractConfig` API

```julia
# ForwardDiff v0.1
ForwardDiff.gradient(f, x; chunk_size = 10)

# ForwardDiff v0.2
ForwardDiff.gradient(f, x, Chunk{10}())

# ForwardDiff v0.3 & v0.4
ForwardDiff.gradient(f, x, ForwardDiff.GradientConfig{10}(x))

# ForwardDiff v0.5 & above
ForwardDiff.gradient(f, x, ForwardDiff.GradientConfig(f, x ForwardDiff.Chunk{10}()))
```

# Enabling Multithreading

```julia
# ForwardDiff v0.1 & v0.2
ForwardDiff.gradient(f, x; multithread = true)

# ForwardDiff v0.3 & v0.4
ForwardDiff.gradient(f, x, ForwardDiff.MultithreadConfig(ForwardDiff.GradientConfig(x)))

# ForwardDiff v0.5 & above
error("ForwardDiff no longer supports internal multithreading.")
```

# Retrieving Lower-Order Results

```julia
# ForwardDiff v0.1
answer, results = ForwardDiff.hessian(f, x, AllResults)
v = ForwardDiff.value(results)
g = ForwardDiff.gradient(results)
h = ForwardDiff.hessian(results) # == answer

# ForwardDiff v0.2
out = HessianResult(x)
ForwardDiff.hessian!(out, f, x)
v = ForwardDiff.value(out)
g = ForwardDiff.gradient(out)
h = ForwardDiff.hessian(out)

# ForwardDiff v0.3 & above
using DiffBase
out = DiffBase.HessianResult(x)
ForwardDiff.hessian!(out, f, x)
v = DiffBase.value(out)
g = DiffBase.gradient(out)
h = DiffBase.hessian(out)

# ForwardDiff v0.6 & above
using DiffResults
out = DiffResults.HessianResult(x)
out = ForwardDiff.hessian!(out, f, x) # re-alias output!
v = DiffResults.value(out)
g = DiffResults.gradient(out)
h = DiffResults.hessian(out)
```

## Higher-Order Differentiation

In order to maintain feature parity between all API functions, ForwardDiff no longer
provides the `tensor` function. Instead, users can take higher-order/higher-dimensional
derivatives by composing existing API functions. For example, here's how to reimplement
`tensor`:

```julia
# ForwardDiff v0.1
ForwardDiff.tensor(f, x)

# ForwardDiff v0.2 & above
function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

tensor(f, x)
```

## Creating Differentiation Functions

Since v0.2, ForwardDiff no longer supports automatic generation of differentiation
functions. Instead, users explicitly define their own functions using ForwardDiff's API.
This leads to clearer code, less "magic", and more flexibility.


```julia
# ForwardDiff v0.1
df = ForwardDiff.derivative(f)

# ForwardDiff v0.2 & above
df = x -> ForwardDiff.derivative(f, x)
```

```julia
# ForwardDiff v0.1
# in-place gradient function of f
gf! = ForwardDiff.gradient(f, mutates = true)

# ForwardDiff v0.2 & above
gf! = (out, x) -> ForwardDiff.gradient!(out, f, x)
```

```julia
# ForwardDiff v0.1
# in-place Jacobian function of f!(y, x):
jf! = ForwardDiff.jacobian(f!, mutates = true, output_length = length(y))

# ForwardDiff v0.2 & above
jf! = (out, y, x) -> ForwardDiff.jacobian!(out, f!, y, x)
```
