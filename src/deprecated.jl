old_new_pairs = [
    (:forwarddiff_gradient, :(gradient_func{N}(f::Function, ::Type{Dim{N}}; mutates=false))),
    (:forwarddiff_gradient!, :(gradient_func{N}(f::Function, ::Type{Dim{N}}; mutates=true))),
    (:forwarddiff_jacobian, :(jacobian_func{N}(f::Function, ::Type{Dim{N}}; mutates=false))),
    (:forwarddiff_jacobian!, :(jacobian_func{N}(f::Function, ::Type{Dim{N}}; mutates=true))),
    (:forwarddiff_hessian, :(hessian_func{N}(f::Function, ::Type{Dim{N}}; mutates=false))),
    (:forwarddiff_hessian!, :(hessian_func{N}(f::Function, ::Type{Dim{N}}; mutates=true))),
    (:forwarddiff_tensor, :(tensor_func{N}(f::Function, ::Type{Dim{N}}; mutates=false))),
    (:forwarddiff_tensor!, :(tensor_func{N}(f::Function, ::Type{Dim{N}}; mutates=true)))
]

for (old_func_name, new_func_sig) in old_new_pairs

    err_str = string("The function:\n",
                     "\t$(old_func_name){T}(f::Function, ::Type{T}; fadtype::Symbol, args...)\n",
                     " is deprecated. Use:\n",
                     "\t$(new_func_sig)\n",
                     " instead of the deprecated function.")
    
    @eval begin
        $(old_func_name)(args...) = error($err_str)
    end
end

export forwarddiff_gradient, 
    forwarddiff_gradient!,
    forwarddiff_jacobian, 
    forwarddiff_jacobian!,
    forwarddiff_hessian, 
    forwarddiff_hessian!,
    forwarddiff_tensor, 
    forwarddiff_tensor!