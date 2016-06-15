[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl) [![Coverage Status](https://coveralls.io/repos/JuliaDiff/ForwardDiff.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiff/ForwardDiff.jl?branch=master)
[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.3.svg)](http://pkg.julialang.org/?pkg=ForwardDiff&ver=0.3)
[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.4.svg)](http://pkg.julialang.org/?pkg=ForwardDiff&ver=0.4)
[![ForwardDiff](http://pkg.julialang.org/badges/ForwardDiff_0.5.svg)](http://pkg.julialang.org/?pkg=ForwardDiff)

**[Go To ForwardDiff's Documentation](http://www.juliadiff.org/ForwardDiff.jl/)**

**Warning**: Please read [this issue](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) before attempting nested differentiation with ForwardDiff.

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

- 12/29/2015: [ForwardDiff v0.1.4 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/4293).

- 12/09/2015: [ForwardDiff v0.1.3 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/4182).

- 10/21/2015: [ForwardDiff v0.1.2 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/3835).

- 9/29/2015: [ForwardDiff v0.1.1 has been tagged](https://github.com/JuliaLang/METADATA.jl/pull/3580).

- 9/3/2015: We're releasing ForwardDiff v.0.1.0. A *lot* has changed since the previous version of the package. The best way to get yourself acquainted with the new API is to read our new [documentation](http://www.juliadiff.org/ForwardDiff.jl/).
