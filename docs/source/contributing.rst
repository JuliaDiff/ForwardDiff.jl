How to Contribute
=================

There are a few fairly easy ways for newcomers to substantially improve ForwardDiff.jl. They all revolve around **writing functions for the** ``ForwardDiffNumber`` **types**. This section provides brief tutorials on how to make these contributions. 

If you're not used to using GitHub, here's the workflow you should use:

1. Fork ForwardDiff.jl
2. Make a new branch on your fork, named after whatever changes you'll be making
3. Apply your code changes to the branch on your fork
4. When you're done, submit a PR to ForwardDiff.jl to merge your fork into ForwardDiff.jl's master branch.

Manually Optimizing Unary Functions
-----------------------------------

1. Pick a function to optimize

2. Write out the function's first, second, and third derivatives, reusing shared sub-expressions where possible

3. Implement the function on the ``ForwardDiffNumber`` types

Implementing New Functions
--------------------------

Via Calculus.jl
+++++++++++++++

Manually
++++++++


