using Base.depwarn

Base.@deprecate forwarddiff_gradient(f::Function, T::DataType; fadtype::Symbol=:dual, args...) ForwardDiff.gradient(f, mutates=false)
Base.@deprecate forwarddiff_jacobian(f::Function, T::DataType; fadtype::Symbol=:dual, args...) ForwardDiff.jacobian(f, mutates=false)
Base.@deprecate forwarddiff_hessian(f::Function, T::DataType; fadtype::Symbol=:dual, args...) ForwardDiff.hessian(f, mutates=false)
Base.@deprecate forwarddiff_tensor(f::Function, T::DataType; fadtype::Symbol=:dual, args...) ForwardDiff.tensor(f, mutates=false)

function depr_inplace_fad(fad_func, f)
    warn("Addendum to the deprecation warning above:\n"*
         "The depr_inplace_fad function is actually only meant to be used to patch over the old API for mutating functions. Instead of:\n"*
         "\tdeprecated_inplace_fad($fad_func, $f)\n"*
         "You should use the following:\n"*
         "\t$fad_func($f, mutates=true)\n"*
         "Be aware that mutating functions created with the new API take in the output array as the 1st argument rather than the 2nd.")
    g! = fad_func(f, mutates=true)
    fad!(x,y) = g!(y,x) # old api mutated second argument, not first
    return fad!
end

Base.@deprecate forwarddiff_gradient!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) depr_inplace_fad(ForwardDiff.gradient, f)
Base.@deprecate forwarddiff_jacobian!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) depr_inplace_fad(ForwardDiff.jacobian, f)
Base.@deprecate forwarddiff_hessian!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) depr_inplace_fad(ForwardDiff.hessian, f)
Base.@deprecate forwarddiff_tensor!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) depr_inplace_fad(ForwardDiff.tensor, f)

export forwarddiff_gradient,
    forwarddiff_gradient!,
    forwarddiff_jacobian,
    forwarddiff_jacobian!,
    forwarddiff_hessian,
    forwarddiff_hessian!,
    forwarddiff_tensor,
    forwarddiff_tensor!,
    deprecated_inplace_fad