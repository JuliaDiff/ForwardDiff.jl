# ForwardDiff

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff **generally outperform non-AD algorithms in both speed and accuracy.**

[Wikipedia's automatic differentiation entry](https://en.wikipedia.org/wiki/Automatic_differentiation) is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

ForwardDiff is a registered Julia package, so it can be installed by running:

```julia
julia> Pkg.add("ForwardDiff")
```

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
