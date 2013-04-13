### Scope of AutoDiff.jl

The aim of this repository is to create a package for performing automatic 
differentiation (AD) in Julia. Automatic differentiation, also known as 
algorithmic differentiation, is a methodology for computing numerically the 
derivatives, also called sensitivities, of a function. The AD algorithms 
rely on the chain rule of calculus. They are used for evaluating
* the derivative of arbitrary (finite) order of a function,
* gradients of multivariable functions,
* Jacobians, 
* Hessians,
* tensor coefficients,
* Taylor coefficients,
* numerical solutions of differential equations.
It is envisaged that the AutoDiff Julia package will provide all the above 
functionality incrementally as its development progresses.

### Advantages of automatic differentiation

Automatic, symbolic and numerical differentiation are three distinct approaches 
to the evaluation of derivatives.

Symbolic differentiation takes formulae of functions as its input and 
implements symbolic rules in order to find formulae for the derivatives of the 
given functions. Its drawback is its slowness of execution.

Numerical differentiation is a broad term referring to the usage of numerical 
analysis for the calculation of derivatives. Some of its techniques are for 
example
* finite differences,
* polynomial interpolation and
* the method of undetermined coefficients.

Numerical differentiation incurs truncation errors.

Automatic differentiation resolves the deficiencies of speed and round-off 
errors.

### Implementation specifications

AD methods are classified as being in forward or reverse mode, depending on 
whether the chain rule is applied from right to left or from left to right, 
respectively. The two modes can be found in literature under the names forward 
or reverse accumulation or propagation too. While the forward mode is easier to 
implement, both traversals of the chain rule have relative advantages. 
Therefore, it is intended to provide implementations of both in the AutoDiff 
Julia package.

Another methodological distinction in AD is the one between source code 
transformation, which involves generating source code for the computation of 
the derivatives, and operator overloading, which overloads the usual arithmetic 
operators in order to perform arithmetic on ordered pairs (called dual 
numbers), with ordinary arithmetic on the first elemnent and first order 
differentiation arithmetic on the second element of the pairs. The preferred 
approach in the AutoDiff package is operator overloading at least at the first 
stage of development, since it is the easiest implementation in Julia.

### Initial Julia code for AD

The initial code of AutoDiff was written by Jonas Rauch and was forwarded to 
the Julia community by Keving Squire. Several members of the community have 
expressed interest in developing the code furthermore. If you are interested in 
contributing, please feel free to do so via a pull request.

Below are the two threads that initiated the creation of the AutoDiff package:

https://groups.google.com/forum/#!msg/julia-dev/tXBR04t31vI/Q30VCKAq8o0J

https://groups.google.com/forum/?fromgroups=#!topic/julia-dev/zQFlX1CGfeo

Furthermore, the `src/diff.jl` file of the simple-mcmc package in

https://github.com/fredo-dedup/simple-mcmc

contains some code for automatic differentiation.
