using Base.depwarn

Base.@deprecate forwarddiff_gradient(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_gradient(f, mutates=false)
Base.@deprecate forwarddiff_gradient!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_gradient(f, mutates=true)
Base.@deprecate forwarddiff_jacobian(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_jacobian(f, mutates=false)
Base.@deprecate forwarddiff_jacobian!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_jacobian(f, mutates=true)
Base.@deprecate forwarddiff_hessian(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_hessian(f, mutates=false)
Base.@deprecate forwarddiff_hessian!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_hessian(f, mutates=true)
Base.@deprecate forwarddiff_tensor(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_tensor(f, mutates=false)
Base.@deprecate forwarddiff_tensor!(f::Function, T::DataType; fadtype::Symbol=:dual, args...) fad_tensor(f, mutates=true)

export forwarddiff_gradient, 
    forwarddiff_gradient!,
    forwarddiff_jacobian, 
    forwarddiff_jacobian!,
    forwarddiff_hessian, 
    forwarddiff_hessian!,
    forwarddiff_tensor, 
    forwarddiff_tensor!