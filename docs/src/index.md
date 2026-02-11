# ForwardDiff

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff **generally outperform non-AD algorithms in both speed and accuracy.**

[Wikipedia's automatic differentiation entry](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

ForwardDiff is a registered Julia package, so it can be installed by running:

```julia-repl
julia> Pkg.add("ForwardDiff")
```

Here's a simple example showing the package in action:

```julia-repl
julia> using ForwardDiff

julia> f(x::Vector) = sin(x[1]) + prod(x[2:end]);  # returns a scalar

julia> x = vcat(pi/4, 2:4)
4-element Vector{Float64}:
 0.7853981633974483
 2.0
 3.0
 4.0

julia> ForwardDiff.gradient(f, x)
4-element Vector{Float64}:
  0.7071067811865476
 12.0
  8.0
  6.0

julia> ForwardDiff.hessian(f, x)
4×4 Matrix{Float64}:
 -0.707107  0.0  0.0  0.0
  0.0       0.0  4.0  3.0
  0.0       4.0  0.0  2.0
  0.0       3.0  2.0  0.0
```

Functions like `f` which map a vector to a scalar are the best case for reverse-mode automatic differentiation,
but ForwardDiff may still be a good choice if `x` is not too large, as it is much simpler.
The best case for forward-mode differentiation is a function which maps a scalar to a vector, like this `g`:

```julia-repl
julia> g(y::Real) = [sin(y), cos(y), tan(y)];  # returns a vector

julia> ForwardDiff.derivative(g, pi/4)
3-element Vector{Float64}:
  0.7071067811865476
 -0.7071067811865475
  1.9999999999999998

julia> ForwardDiff.jacobian(x) do x  # anonymous function, returns a length-2 vector
         [sin(x[1]), prod(x[2:end])]
       end
2×4 Matrix{Float64}:
 0.707107   0.0  0.0  0.0
 0.0       12.0  8.0  6.0
```

If you find ForwardDiff useful in your work, we kindly request that you cite [our paper](https://arxiv.org/abs/1607.07892). The relevant [BibLaTex is available in ForwardDiff's README](https://github.com/JuliaDiff/ForwardDiff.jl#publications) (not included here because BibLaTex doesn't play nice with Documenter/Jekyll).
