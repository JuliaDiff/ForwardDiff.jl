Performance Tips
================

Basically, one should heed the `performance tips`_ provided in the Julia docs. The most important tips listed there revolve around the notion of `type stability`_; **always try to make sure that the target function you're passing to ForwardDiff.jl is type-stable.** Type instability in the target function can cause slowness, and in some cases, errors. 

.. _`performance tips`: http://julia.readthedocs.org/en/latest/manual/performance-tips/
.. _`type stability`: http://julia.readthedocs.org/en/latest/manual/performance-tips/#write-type-stable-functions
