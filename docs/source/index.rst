ForwardDiff.jl
==============

ForwardDiff.jl implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff.jl **generally outperform non-AD algorithms in both speed and accuracy.**

This `wikipedia page`_ on automatic differentiation is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as `finite differencing`_). 

For now, ForwardDiff.jl only supports differentiation of functions involving ``T<:Real`` numbers, but we believe extension to ``T<:Complex`` numbers is possible.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   install.rst
   perf_diff.rst
   chunk_vec_modes.rst
   lower_order_results.rst
   caching.rst
   perf_tips.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

TODO

.. _`wikipedia page`: https://en.wikipedia.org/wiki/Automatic_differentiation
.. _`finite differencing`: https://en.wikipedia.org/wiki/Numerical_differentiation