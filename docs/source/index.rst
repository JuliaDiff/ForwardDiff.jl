`Go to ForwardDiff.jl on GitHub`_

ForwardDiff.jl
==============

ForwardDiff.jl implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff.jl **generally outperform non-AD algorithms in both speed and accuracy.**

This `wikipedia page`_ on automatic differentiation is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as `finite differencing`_). 

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   install.rst
   perf_diff.rst
   chunk_vec_modes.rst
   lower_order_results.rst
   caching.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   how_it_works.rst
   types.rst
   contributing.rst
   why_julia.rst

.. _`Go to ForwardDiff.jl on GitHub`: https://github.com/JuliaDiff/ForwardDiff.jl
.. _`wikipedia page`: https://en.wikipedia.org/wiki/Automatic_differentiation
.. _`finite differencing`: https://en.wikipedia.org/wiki/Numerical_differentiation