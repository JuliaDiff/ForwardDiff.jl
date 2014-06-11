[![Build Status](https://travis-ci.org/JuliaDiff/ForwardDiff.jl.png)](https://travis-ci.org/JuliaDiff/ForwardDiff.jl)

ForwardDiff.jl
================================================================================

The `ForwardDiff` package provides an implementation of forward-mode automatic differentiation (FAD) in Julia.

`ForwardDiff` is undergoing development. It currently implements and will include:

1. FAD of gradients, Jacobians, Hessians and tensors, i.e. up to third-order derivatives of univariate and multivariate
functions. This feature is available.

2. FAD of matrix expressions. This feature will be added in the future.

3. A range of different FAD implementations, each with varying racing merits such as range of applicability and
efficiency. Two FAD approaches are available, one of which is type-based and one based on dual numbers. Two more
FAD approaches will be provided, one using the box product for matrices and another one using power series.

4. A unified API across the various FAD methods. The API is operational. It will be kept up-to-date whenever new 
FAD algorithms are added to the package.

Please refer to [the package documentation](http://forwarddiffjl.readthedocs.org/en/latest/) for details.
