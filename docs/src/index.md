# ForwardDiff

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff **generally outperform non-AD algorithms in both speed and accuracy.**

[Wikipedia's automatic differentiation entry](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

ForwardDiff is a registered Julia package, so it can be installed by running:

```julia
julia> Pkg.add("ForwardDiff")
```

If you find ForwardDiff useful in your work, we kindly request that you cite [our paper](https://arxiv.org/abs/1607.07892). The relevant [BibLaTex is available in ForwardDiff's README](https://github.com/JuliaDiff/ForwardDiff.jl#publications) (not included here because BibLaTex doesn't play nice with Documenter/Jekyll).
