Why Julia?
==========

If you're a newcomer to Julia, you may ask "What makes Julia so ideal for forward mode AD methods?" There are several good answers, but the main reason that Julia is desirable is it's **multiple dispatch** feature.

Unlike many other languages, **Julia's type-based operator overloading is fast and natural**, and one of the central design tenets of the langauge. This is because Julia is a dynamic, JIT-compiled language - the compiled bytecode of a Julia function is tied directly to the types with which the function is called, so **the compiler can optimize every Julia method for the specific input type at runtime**.

Here's a (somewhat contrived) example that illustrates the flexibility of multiple dispatch:

.. code-block:: julia

    julia> f(x) = sqrt(x)
    f (generic function with 1 method)

    julia> f(2) # method f(::Int64) is JIT-compiled and called
    1.4142135623730951

    julia> f(1+2im) # new method f(::Complex{Int64}) is JIT-compiled and called
    1.272019649514069 + 0.7861513777574233im

    julia> f(x::AbstractString) = "√$x" # if called on some sort of string, do this instead
    f (generic function with 2 methods)

    julia> f("2") # new method f(::ASCIIString) is JIT-compiled and called
    "√2"

    julia> f(2) # call the version of f(::Int64) we already compiled
    1.4142135623730951

So finally, to answer the original question: **Julia's multiple dispatch allows us to efficiently overload core Julia methods (e.g.** ``sin``, ``lgamma``, ``log``, **etc.) on** ``ForwardDiffNumber`` **types to accumulate derivative information.**