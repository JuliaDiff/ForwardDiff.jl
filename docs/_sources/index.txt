`Go to ForwardDiff on GitHub`_

ForwardDiff.jl
==============

ForwardDiff implements methods to take **derivatives**, **gradients**, **Jacobians**, **Hessians**, and higher-order derivatives of native Julia functions (or any callable object, really) using **forward mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff **generally outperform non-AD algorithms in both speed and accuracy.**

This `wikipedia page`_ on automatic differentiation is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as `finite differencing`_).

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   install.rst
   limitations.rst
   api.rst
   chunk.rst
   lower_order_results.rst
   upgrade.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   how_it_works.rst
   types.rst
   contributing.rst

Publications
------------

If you find ForwardDiff useful in your work, we kindly request that you cite `the following paper`_::

    @article{RevelsLubinPapamarkou2016,
       title = {Forward-Mode Automatic Differentiation in Julia},
      author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},
     journal = {arXiv:1607.07892 [cs.MS]},
        year = {2016},
        url = {https://arxiv.org/abs/1607.07892}
    }

.. _`Go to ForwardDiff on GitHub`: https://github.com/JuliaDiff/ForwardDiff.jl
.. _`wikipedia page`: https://en.wikipedia.org/wiki/Automatic_differentiation
.. _`finite differencing`: https://en.wikipedia.org/wiki/Numerical_differentiation
.. _`the following paper`: https://arxiv.org/abs/1607.07892
