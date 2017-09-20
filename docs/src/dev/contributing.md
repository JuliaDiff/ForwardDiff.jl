# How to Contribute

There are a few fairly easy ways for newcomers to substantially improve ForwardDiff, and
they all revolve around **writing functions for** `Dual` **numbers**. This section
provides brief tutorials on how to make these contributions.

If you're new GitHub, here's an outline of the workflow you should use:

1. Fork ForwardDiff
2. Make a new branch on your fork, named after whatever changes you'll be making
3. Apply your code changes to the branch on your fork
4. When you're done, submit a PR to ForwardDiff to merge your fork into ForwardDiff's master branch.

## Adding New Derivative Definitions

In general, new derivative implementations for `Dual` are automatically defined via simple
symbolic rules. ForwardDiff accomplishes this by looping over the rules provided by
[the DiffRules package](https://github.com/JuliaDiff/DiffRules.jl) and using them to
auto-generate `Dual` definitions. Conveniently, these auto-generated definitions are also
automatically tested.

Thus, in order to add a new derivative implementation for `Dual`, you should define the
appropriate derivative rule(s) in DiffRules, and then check that calling the function on
`Dual` instances delivers the desired result.

Depending on your function, ForwardDiff's auto-definition mechanism might need to be
expanded to support it. If this is the case, file an issue/PR so that ForwardDiff's
maintainers can help you out.
