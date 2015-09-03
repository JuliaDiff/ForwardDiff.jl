Caching Options
===============

When Does Caching Matter?
-------------------------

If you're going to be repeatedly evaluating the gradient/Hessian/etc. of a function, it's probably worth it to generate the corresponding method beforehand rather than call ``hessian(f, x)`` a bunch of times. For example, this:
    
.. code-block:: julia

    inputs = [rand(100) for i in 1:100]
    for x in inputs
        hess = hessian(f, x)
        ... # do something with hess
    end

...should really be written like this:
    
.. code-block:: julia

    h = hessian(f) # generate H(f) first
    inputs = [rand(100) for i in 1:100]
    for x in inputs
        hess = h(x)
        ... # do something with hess
    end

The latter style is preferrable because ``hessian(f, x)`` requires the creation of various temporary "work arrays" to perform the calculation. Generating ``h = hessian(f)`` beforehand allows the temporary arrays to be cached in subsequent calls to ``h``, saving both time and memory over the course of the loop.

Manual Caching
--------------

This caching can be handled "manually", if one wishes, by utilizing the provided ``ForwardDiffCache`` type and the ``cache`` keyword argument:

.. code-block:: julia

    my_cache = ForwardDiffCache() # make new cache to pass in to our function
    inputs = [rand(1000) for i in 1:100]
    for x in inputs
        # just as efficient as pre-generating h, because it can reuse my_cache
        hess = hessian(f, x, cache=my_cache) 
        ... # do something with hess
    end

The above case becomes more realistically useful if you consider a situation in which ``f`` was variable throughout the loop - you'd still want to avoid wasting time/memory creating work vectors, but you wouldn't want to have to re-generate ``hessian(f)`` over and over. Manually managing the cache provides a solution to this problem.

Note that the ``cache`` keyword argument is supported for all of ForwardDiff.jl's differentiation methods.
