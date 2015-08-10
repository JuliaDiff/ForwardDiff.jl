[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.svg?branch=api-refactor)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl) [![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=api-refactor&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=api-refactor)

# ForwardDiff.jl

The `ForwardDiff` package provides a type-based implementation of forward mode automatic differentiation (FAD) in Julia. [The wikipedia page on automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of FAD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

## What can I do with this package?

This package contains methods to efficiently take derivatives, Jacobians, and Hessians of native Julia functions (or any callable object, really). While performance varies depending on the functions you evaluate, this package generally outperforms non-AD methods in memory usage, speed, and accuracy.

A third-order generalization of the Hessian is also implemented (see `tensor` below). 

For now, we only support for functions involving `T<:Real`s, but we believe extension to numbers of type `T<:Complex` is possible.

## Usage

---
#### Derivative of `f(x::Number) → Number` or `f(x::Number) → Array`
---

- **`derivative!(output::Array, f, x::Number)`**
    
    Compute `f'(x)`, storing the output in `output`.

- **`derivative(f, x::Number)`**
    
    Compute `f'(x)`.

- **`derivative(f; mutates=false)`**
    
    Return the function `f'`. If `mutates=false`, then the returned function has the form `derivf(x) -> derivative(f, x)`. If `mutates = true`, then the returned function has the form `derivf!(output, x) -> derivative!(output, f, x)`.

---
#### Gradient of `f(x::Vector) → Number`
---

- **`gradient!(output::Vector, f, x::Vector)`**

    Compute `∇f(x)`, storing the output in `output`.

- **`ForwardDiff.gradient{T}(f, x::Vector{T})`**

    Compute `∇f(x)`, where `T` is the element type of both the input and output. `ForwardDiff` must be used as a qualifier when calling `gradient` to avoid conflict with `Base.gradient`.

- **`ForwardDiff.gradient(f; mutates=false)`**

    Return the function `∇f`. If `mutates=false`, then the returned function has the form `gradf(x) -> gradient(f, x)`. If `mutates = true`, then the returned function has the form `gradf!(output, x) -> gradient!(output, f, x)`. By default, `mutates` is set to `false`. `ForwardDiff` must be used as a qualifier when calling `gradient` to avoid conflict with `Base.gradient`.

---
#### Jacobian of `f(x:Vector) → Vector`
---

- **`jacobian!(output::Matrix, f, x::Vector)`**

    Compute `J(f(x))`, storing the output in `output`.

- **`jacobian{T}(f, x::Vector{T})`**

    Compute `J(f(x))`, where `T` is the element type of both the input and output.

- **`jacobian(f; mutates=false)`**

    Return the function `J(f)`. If `mutates=false`, then the returned function has the form `jacf(x) -> jacobian(f, x)`. If `mutates = true`, then the returned function has the form `jacf!(output, x) -> jacobian!(output, f, x)`. By default, `mutates` is set to `false`.

---
#### Hessian of `f(x::Vector) → Number`
---

- **`hessian!(output::Matrix, f, x::Vector)`**

    Compute `H(f(x))`, storing the output in `output`.

- **`hessian{T}(f, x::Vector{T})`**

    Compute `H(f(x))`, where `T` is the element type of both the input and output.

- **`hessian(f; mutates=false)`**

    Return the function `H(f)`. If `mutates=false`, then the returned function has the form `hessf(x) -> hessian(f, x, S)`. If `mutates = true`, then the returned function has the form `hessf!(output, x) -> hessian!(output, f, x)`. By default, `mutates` is set to `false`.

---
#### Third-order Taylor series term of `f(x::Vector) → Number`
---

[This Math StackExchange post](http://math.stackexchange.com/questions/556951/third-order-term-in-taylor-series) actually has an answer that explains this term fairly clearly.

- **`tensor!{T}(output::Array{T,3}, f, x::Vector)`**

    Compute `∑D³f(x)`, storing the output in `output`.

- **`tensor{T}(f, x::Vector{T})`**

    Compute `∑D³f(x)`, where `T` is the element type of both the input and output.

- **`tensor(f; mutates=false)`**

    Return the function ``∑D³f``. If `mutates=false`, then the returned function has the form `tensf(x) -> tensor(f, x)`. If `mutates = true`, then the returned function has the form `tensf!(output, x) -> tensor!(output, f, x)`. By default, `mutates` is set to `false`.
