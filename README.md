[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl) [![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=master)

[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.3.svg)](http://pkg.julialang.org/?pkg=ForwardDiff&ver=0.3)
[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.4.svg)](http://pkg.julialang.org/?pkg=ForwardDiff&ver=0.4)
[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.5.svg)](http://pkg.julialang.org/?pkg=ForwardDiff)
[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.6.svg)](http://pkg.julialang.org/?pkg=ForwardDiff)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](http://www.juliadiff.org/ForwardDiff.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](http://www.juliadiff.org/ForwardDiff.jl/latest)

# ForwardDiff.jl

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff **generally outperform non-AD algorithms in both speed and accuracy.**

Here's a simple example showing the package in action:

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

julia> g = x -> ForwardDiff.gradient(f, x); # g = ∇f

julia> g(x)
5-element Array{Float64,1}:
 1.01358
 2.50014
 1.72574
 1.10139
 1.2445

julia> ForwardDiff.hessian(f, x)
5x5 Array{Float64,2}:
 0.585111  3.48083  1.7706    0.994057  1.03257
 3.48083   1.06079  5.79299   3.25245   3.37871
 1.7706    5.79299  0.423981  1.65416   1.71818
 0.994057  3.25245  1.65416   0.251396  0.964566
 1.03257   3.37871  1.71818   0.964566  0.140689
 ```

## News

- 3/2017: [ForwardDiff v0.4.0 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/8209). There are no API changes, but [support for Julia v0.4 has been dropped](https://github.com/JuliaDiff/ForwardDiff.jl/pull/195) moving forward.

- 12/2016: [ForwardDiff v0.3.0 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/7177). See our [upgrade guide](http://www.juliadiff.org/ForwardDiff.jl/upgrade.html) for details regarding user-facing changes.

- 6/2016: [ForwardDiff v0.2.0 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/5389). See our [upgrade guide](http://www.juliadiff.org/ForwardDiff.jl/upgrade.html) for details regarding user-facing changes.

- 9/2015: We're releasing ForwardDiff v.0.1.0. A *lot* has changed since the previous version of the package. The best way to get yourself acquainted with the new API is to read our new [documentation](http://www.juliadiff.org/ForwardDiff.jl/).

## Publications

If you find ForwardDiff useful in your work, we kindly request that you cite [the following paper](https://arxiv.org/abs/1607.07892):

```
@article{RevelsLubinPapamarkou2016,
    title = {Forward-Mode Automatic Differentiation in Julia},
   author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},
  journal = {arXiv:1607.07892 [cs.MS]},
     year = {2016},
     url = {https://arxiv.org/abs/1607.07892}
}
```
