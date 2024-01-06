# Differentiation API

```@meta
CurrentModule = ForwardDiff
```

## Derivatives of `f(x::Real)::Union{Real,AbstractArray}`

```@docs
ForwardDiff.derivative
ForwardDiff.derivative!
```

## Gradients of `f(x::AbstractArray)::Real`

```@docs
ForwardDiff.gradient
ForwardDiff.gradient!
```

## Jacobians of `f(x::AbstractArray)::AbstractArray`

```@docs
ForwardDiff.jacobian
ForwardDiff.jacobian!
```

## Hessians of `f(x::AbstractArray)::Real`

```@docs
ForwardDiff.hessian
ForwardDiff.hessian!
```

## Preallocating/Configuring Work Buffers

For the sake of convenience and performance, all "extra" information used by ForwardDiff's
API methods is bundled up in the `ForwardDiff.AbstractConfig` family of types. These types
allow the user to easily feed several different parameters to ForwardDiff's API methods,
such as chunk size, work buffers, and perturbation seed configurations.

ForwardDiff's basic API methods will allocate these types automatically by default,
but you can drastically reduce memory usage if you preallocate them yourself.

Note that for all constructors below, the chunk size `N` may be explicitly provided,
or omitted, in which case ForwardDiff will automatically select a chunk size for you.
However, it is highly recommended to specify the chunk size manually when possible
(see [Configuring Chunk Size](@ref)).

Note also that configurations constructed for a specific function `f` cannot be reused to
differentiate other functions (though can be reused to differentiate `f` at different
values). To construct a configuration which can be reused to differentiate any function, you
can pass `nothing` as the function argument. While this is more flexible, it decreases
ForwardDiff's ability to catch and prevent [perturbation
confusion](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83).

```@docs
ForwardDiff.DerivativeConfig
ForwardDiff.GradientConfig
ForwardDiff.JacobianConfig
ForwardDiff.HessianConfig
```
