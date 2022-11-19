[![CI](https://github.com/JuliaDiff/ForwardDiff.jl/workflows/CI/badge.svg)](https://github.com/JuliaDiff/ForwardDiff.jl/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=master)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/ForwardDiff.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/ForwardDiff.jl/dev)

# ForwardDiff.jl

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff generally outperform non-AD algorithms (such as finite-differencing) in both speed and accuracy.

Here's a simple example showing the package in action:

```julia
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

```julia
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

See [ForwardDiff's documentation](https://juliadiff.org/ForwardDiff.jl/stable) for full details on how to use this package.
ForwardDiff relies on [DiffRules](https://github.com/JuliaDiff/DiffRules.jl) for the derivatives of many simple function such as `sin`.

See the [JuliaDiff web page](https://juliadiff.org) for other automatic differentiation packages.

## Publications

If you find ForwardDiff useful in your work, we kindly request that you cite [the following paper](https://arxiv.org/abs/1607.07892):

```bibtex
@article{RevelsLubinPapamarkou2016,
    title = {Forward-Mode Automatic Differentiation in {J}ulia},
   author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},
  journal = {arXiv:1607.07892 [cs.MS]},
     year = {2016},
      url = {https://arxiv.org/abs/1607.07892}
}
```
