[![Build Status](https://travis-ci.org/scidom/ForwardDiff.jl.png)](https://travis-ci.org/scidom/ForwardDiff.jl)

## Overview of package's scope

The `ForwardDiff` package provides an implementation of forward mode automatic differentiation (FAD) in Julia.

The package is undergoing development. Four different approaches for performing FAD will be made available
progressively:

1. FAD using dual numbers. This approach has been already implemented.

2. FAD using generalized dual numbers. Dual numbers can be generalized to facilitate the exact computation of higher
order derivatives. [This](http://jliszka.github.io/2013/10/24/exact-numeric-nth-derivatives.html) and
[this](http://duaeliststudios.com/automatic-differentiation-with-dual-numbers/) blog post give an idea of the
prospective implementation, which will probably make use of the `Polynomial` package.

3. FAD of matrix functions based on [this](http://link.springer.com/chapter/10.1007%2F978-3-642-30023-3_7) publication.

4. FAD using a number of Julia types introduced solely for the purpose of performing automatic differentiation. This is
in a sense the most archaic approach; it offers yet another FAD implementation, useful for benchmarking purposes
and as an alternative tool for the user. This approach allows already to evaluate gradients, Jacobians and Hessians. It
will be extended to support tensors.
