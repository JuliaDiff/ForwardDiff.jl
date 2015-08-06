[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.svg?branch=nduals-refactor)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl)

[![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=nduals-refactor&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=nduals-refactor)

# ForwardDiff.jl

The `ForwardDiff` package provides a type-based implementation of forward mode automatic differentiation (FAD) in Julia. [The wikipedia page on automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of FAD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

## What can I do with this package?

This package contains methods to efficiently take derivatives, Jacobians, and Hessians of native Julia functions (or any callable object, really). While performance varies depending on the functions you evaluate, this package generally outperforms non-AD methods in memory usage, speed, and accuracy.

A third-order generalization of the Hessian is also implemented (see `tensor` below). 

For now, we only support for functions involving `T<:Real`s, but we believe extension to numbers of type `T<:Complex` is possible.

## Usage

---
#### Derivative of `f: R → R` or `f: R → Rᵐ¹ × Rᵐ² × ⋯ × Rᵐⁱ`
---

- **`derivative!(f, x::Number, output::Array)`**
    
    Compute `f'(x)`, storing the output in `output`.

- **`derivative(f, x::Number)`**
    
    Compute `f'(x)`.

- **`derivative(f; mutates=false)`**
    
    Return the function `f'`. If `mutates=false`, then the returned function has the form `derivf(x) -> derivative(f, x)`. If `mutates = true`, then the returned function has the form `derivf!(x, output) -> derivative!(f, x, output)`.

---
#### Gradient of `f: Rⁿ → R`
---

- **`gradient!(f, x::Vector, output::Vector)`**

    Compute `∇f(x)`, storing the output in `output`.

- **`gradient{T,S}(f, x::Vector{T}, ::Type{S}=T)`**

    Compute `∇f(x)`, where `S` is the element type of the output. By default, `S` is set to the element type of the input (`T`).

- **`gradient(f; mutates=false)`**

    Return the function `∇f`. If `mutates=false`, then the returned function has the form `gradf(x, ::Type{S}=T) -> gradient(f, x, S)`. If `mutates = true`, then the returned function has the form `gradf!(x, output) -> gradient!(f, x, output)`. By default, `mutates` is set to `false`.

---
#### Jacobian of `f: Rⁿ → Rᵐ`
---

- **`jacobian!(f, x::Vector, output::Matrix)`**

    Compute `J(f(x))`, storing the output in `output`.

- **`jacobian{T,S}(f, x::Vector{T}, ::Type{S}=T)`**

    Compute `J(f(x))`, where `S` is the element type of the output. By default, `S` is set to the element type of the input (`T`).

- **`jacobian(f; mutates=false)`**

    Return the function `J(f)`. If `mutates=false`, then the returned function has the form `jacf(x, ::Type{S}=T) -> jacobian(f, x, S)`. If `mutates = true`, then the returned function has the form `jacf!(x, output) -> jacobian!(f, x, output)`. By default, `mutates` is set to `false`.

---
#### Hessian of `f: Rⁿ → R`
---

- **`hessian!(f, x::Vector, output::Matrix)`**

    Compute `H(f(x))`, storing the output in `output`.

- **`hessian{T,S}(f, x::Vector{T}, ::Type{S}=T)`**

    Compute `H(f(x))`, where `S` is the element type of the output. By default, `S` is set to the element type of the input (`T`).

- **`hessian(f; mutates=false)`**

    Return the function `H(f)`. If `mutates=false`, then the returned function has the form `hessf(x, ::Type{S}=T) -> hessian(f, x, S)`. If `mutates = true`, then the returned function has the form `hessf!(x, output) -> hessian!(f, x, output)`. By default, `mutates` is set to `false`.

---
#### Third-order Taylor series term of `f: Rⁿ → R`
---

[This Math StackExchange post](http://math.stackexchange.com/questions/556951/third-order-term-in-taylor-series) actually has an answer that explains this term fairly clearly.

- **`tensor!{S}(f, x::Vector, output::Array{S,3})`**

    Compute `∑D³f(x)`, storing the output in `output`.

- **`tensor{T,S}(f, x::Vector{T}, ::Type{S}=T)`**

    Compute `∑D³f(x)`, where `S` is the element type of the output. By default, `S` is set to the element type of the input (`T`).

- **`tensor(f; mutates=false)`**

    Return the function ``∑D³f``. If `mutates=false`, then the returned function has the form `tensf(x, ::Type{S}=T) -> tensor(f, x, S)`. If `mutates = true`, then the returned function has the form `tensf!(x, output) -> tensor!(f, x, output)`. By default, `mutates` is set to `false`.
