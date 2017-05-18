# How ForwardDiff Works

ForwardDiff is an implementation of [forward mode automatic
differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) in
Julia. There are two key components of this implementation: the `Dual` type, and the API.

## Dual Number Implementation

Partial derivatives are stored in the `Partials` type:


```julia
struct Partials{N,V} <: AbstractVector{V}
    values::NTuple{N,V}
end
```

The `Partials` type is used to implement the `Dual` type:

```julia
struct Dual{T,V<:Real,N} <: Real
    value::V
    partials::Partials{N,V}
end
```

This type represents an `N`-dimensional [dual number](https://en.wikipedia.org/wiki/Dual_number)
coupled with a tag parameter `T` in order to prevent [perturbation
confusion](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83). This dual number
type is implemented to have the following mathematical behavior:

```math
f(a + \sum_{i=1}^N b_i \epsilon_i) = f(a) + f'(a) \sum_{i=1}^N b_i \epsilon_i
```

where the ``a`` component is stored in the `value` field and the ``b``
components are stored in the `partials` field. This property of dual numbers is the
central feature that allows ForwardDiff to take derivatives.

In order to implement the above property, elementary numerical functions on a `Dual`
number are overloaded to evaluate both the original function, *and* evaluate the derivative
of the function, propogating the derivative via multiplication. For example, `Base.sin`
can be overloaded on `Dual` like so:

```julia
Base.sin(d::Dual{T}) where {T} = Dual{T}(sin(value(d)), cos(value(d)) * partials(d))
```

If we assume that a general function `f` is composed of entirely of these elementary
functions, then the chain rule enables our derivatives to compose as well. Thus, by
overloading a plethora of elementary functions, we can differentiate generic functions
composed of them by passing in a `Dual` number and looking at the output.

We won't discuss higher-order differentiation in detail, but the reader is encouraged to
learn about [hyper-dual numbers](https://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf),
which extend dual numbers to higher orders by introducing extra ``\epsilon`` terms that can
cross-multiply. ForwardDiff's `Dual` number implementation naturally supports hyper-dual
numbers without additional code by allowing instances of the `Dual` type to nest within each
other. For example, a second-order hyper-dual number has the type `Dual{T,Dual{S,V,M},N}`, a
third-order hyper-dual number has the type `Dual{T,Dual{S,Dual{R,V,K},M},N}`, and so on.

## ForwardDiff's API

The second component provided by this package is the API, which abstracts away the number
types and makes it easy to execute familiar calculations like gradients and Hessians. This
way, users don't have to understand `Dual` numbers in order to make use of the package.

The job of the API functions is to performantly seed input values with `Dual` numbers,
pass the seeded value into the target function, and extract the derivative information from
the result. For example, to calculate the partial derivatives for the gradient of a function
``f`` at an input vector ``\vec{x}``, we would do the following:

```math
\vec{x} = \begin{bmatrix}
               x_1 \\
               \vdots \\
               x_i \\
               \vdots \\
               x_N
           \end{bmatrix}
\to
\vec{x}_{\epsilon} = \begin{bmatrix}
                         x_1 + \epsilon_1 \\
                         \vdots \\
                         x_i + \epsilon_i \\
                         \vdots \\
                         x_N + \epsilon_N
                     \end{bmatrix}
\to
f(\vec{x}_{\epsilon}) = f(\vec{x}) + \sum_{i=1}^N \frac{\delta f(\vec{x})}{\delta x_i} \epsilon_i
```

In reality, ForwardDiff does this calculation in chunks of the input vector (see
[Configuring Chunk Size](@ref) for details). To provide a simple example of this, let's
examine the case where the input vector size is 4 and the chunk size is 2. It then takes
two calls to ``f`` to evaluate the gradient:

```math
\vec{x} = \begin{bmatrix}
               x_1 \\
               x_2 \\
               x_3 \\
               x_4
           \end{bmatrix}

\vec{x}_{\epsilon} = \begin{bmatrix}
                        x_1 + \epsilon_1 \\
                        x_2 + \epsilon_2 \\
                        x_3 \\
                        x_4
                     \end{bmatrix}
\to
f(\vec{x}_{\epsilon}) = f(\vec{x}) + \frac{\delta f(\vec{x})}{\delta x_1} \epsilon_1 + \frac{\delta f(\vec{x})}{\delta x_2} \epsilon_2

\vec{x}_{\epsilon} = \begin{bmatrix}
                        x_1 \\
                        x_2 \\
                        x_3 + \epsilon_1 \\
                        x_4 + \epsilon_2
                     \end{bmatrix}
\to
f(\vec{x}_{\epsilon}) = f(\vec{x}) + \frac{\delta f(\vec{x})}{\delta x_3} \epsilon_1 + \frac{\delta f(\vec{x})}{\delta x_4} \epsilon_2
```

This seeding process is similar for Jacobians, so we won't rehash it here.
