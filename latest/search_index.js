var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#ForwardDiff-1",
    "page": "Introduction",
    "title": "ForwardDiff",
    "category": "section",
    "text": "ForwardDiff implements methods to take derivatives, gradients, Jacobians, Hessians, and higher-order derivatives of native Julia functions (or any callable object, really) using forward mode automatic differentiation (AD).While performance can vary depending on the functions you evaluate, the algorithms implemented by ForwardDiff generally outperform non-AD algorithms in both speed and accuracy.Wikipedia's automatic differentiation entry is a useful resource for learning about the advantages of AD techniques over other common differentiation methods (such as finite differencing).ForwardDiff is a registered Julia package, so it can be installed by running:julia> Pkg.add(\"ForwardDiff\")If you find ForwardDiff useful in your work, we kindly request that you cite the following paper:@article{RevelsLubinPapamarkou2016,\n   title = {Forward-Mode Automatic Differentiation in Julia},\n  author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},\n journal = {arXiv:1607.07892 [cs.MS]},\n    year = {2016},\n    url = {https://arxiv.org/abs/1607.07892}\n}"
},

{
    "location": "user/limitations.html#",
    "page": "Limitations of ForwardDiff",
    "title": "Limitations of ForwardDiff",
    "category": "page",
    "text": ""
},

{
    "location": "user/limitations.html#Limitations-of-ForwardDiff-1",
    "page": "Limitations of ForwardDiff",
    "title": "Limitations of ForwardDiff",
    "category": "section",
    "text": "ForwardDiff works by injecting user code with new number types that collect derivative information at runtime. Naturally, this technique has some limitations. Here's a list of all the roadblocks we've seen users run into (\"target function\" here refers to the function being differentiated):The target function can only be composed of generic Julia functions. ForwardDiff cannot propagate derivative information through non-Julia code. Thus, your function may not work if it makes calls to external, non-Julia programs, e.g. uses explicit BLAS calls instead of Ax_mul_Bx-style functions.\nThe target function must be unary (i.e., only accept a single argument). ForwardDiff.jacobian is an exception to this rule.\nThe target function must be written generically enough to accept numbers of type T<:Real as input (or arrays of these numbers). The function doesn't require a specific type signature, as long as the type signature is generic enough to avoid breaking this rule. This also means that any storage assigned used within the function must be generic as well (see this comment for an example).\nThe types of array inputs must be subtypes of AbstractArray . Non-AbstractArray array-like types are not officially supported."
},

{
    "location": "user/api.html#",
    "page": "Differentiation API",
    "title": "Differentiation API",
    "category": "page",
    "text": ""
},

{
    "location": "user/api.html#Differentiation-API-1",
    "page": "Differentiation API",
    "title": "Differentiation API",
    "category": "section",
    "text": "CurrentModule = ForwardDiff"
},

{
    "location": "user/api.html#ForwardDiff.derivative",
    "page": "Differentiation API",
    "title": "ForwardDiff.derivative",
    "category": "Function",
    "text": "ForwardDiff.derivative(f, x::Real)\n\nReturn df/dx evaluated at x, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), Union{Real,AbstractArray}).\n\n\n\nForwardDiff.derivative(f!, y::AbstractArray, x::Real, cfg::DerivativeConfig = DerivativeConfig(f!, y, x))\n\nReturn df!/dx evaluated at x, assuming f! is called as f!(y, x) where the result is stored in y.\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.derivative!",
    "page": "Differentiation API",
    "title": "ForwardDiff.derivative!",
    "category": "Function",
    "text": "ForwardDiff.derivative!(result::Union{AbstractArray,DiffResult}, f, x::Real)\n\nCompute df/dx evaluated at x and store the result(s) in result, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), Union{Real,AbstractArray}).\n\n\n\nForwardDiff.derivative!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::Real, cfg::DerivativeConfig = DerivativeConfig(f!, y, x))\n\nCompute df!/dx evaluated at x and store the result(s) in result, assuming f! is called as f!(y, x) where the result is stored in y.\n\n\n\n"
},

{
    "location": "user/api.html#Derivatives-of-f(x::Real)::Union{Real,AbstractArray}-1",
    "page": "Differentiation API",
    "title": "Derivatives of f(x::Real)::Union{Real,AbstractArray}",
    "category": "section",
    "text": "ForwardDiff.derivative\nForwardDiff.derivative!"
},

{
    "location": "user/api.html#ForwardDiff.gradient",
    "page": "Differentiation API",
    "title": "ForwardDiff.gradient",
    "category": "Function",
    "text": "ForwardDiff.gradient(f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x))\n\nReturn ∇f evaluated at x, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), Real).\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.gradient!",
    "page": "Differentiation API",
    "title": "ForwardDiff.gradient!",
    "category": "Function",
    "text": "ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x))\n\nCompute ∇f evaluated at x and store the result(s) in result, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), Real).\n\n\n\n"
},

{
    "location": "user/api.html#Gradients-of-f(x::AbstractArray)::Real-1",
    "page": "Differentiation API",
    "title": "Gradients of f(x::AbstractArray)::Real",
    "category": "section",
    "text": "ForwardDiff.gradient\nForwardDiff.gradient!"
},

{
    "location": "user/api.html#ForwardDiff.jacobian",
    "page": "Differentiation API",
    "title": "ForwardDiff.jacobian",
    "category": "Function",
    "text": "ForwardDiff.jacobian(f, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f, x))\n\nReturn J(f) evaluated at x, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), AbstractArray).\n\n\n\nForwardDiff.jacobian(f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f!, y, x))\n\nReturn J(f!) evaluated at x,  assuming f! is called as f!(y, x) where the result is stored in y.\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.jacobian!",
    "page": "Differentiation API",
    "title": "ForwardDiff.jacobian!",
    "category": "Function",
    "text": "ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f, x))\n\nCompute J(f) evaluated at x and store the result(s) in result, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), AbstractArray).\n\n\n\nForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f!, y, x))\n\nCompute J(f!) evaluated at x and store the result(s) in result, assuming f! is called as f!(y, x) where the result is stored in y.\n\nThis method assumes that isa(f(x), AbstractArray).\n\n\n\n"
},

{
    "location": "user/api.html#Jacobians-of-f(x::AbstractArray)::AbstractArray-1",
    "page": "Differentiation API",
    "title": "Jacobians of f(x::AbstractArray)::AbstractArray",
    "category": "section",
    "text": "ForwardDiff.jacobian\nForwardDiff.jacobian!"
},

{
    "location": "user/api.html#ForwardDiff.hessian",
    "page": "Differentiation API",
    "title": "ForwardDiff.hessian",
    "category": "Function",
    "text": "ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x))\n\nReturn H(f) (i.e. J(∇(f))) evaluated at x, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), Real).\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.hessian!",
    "page": "Differentiation API",
    "title": "ForwardDiff.hessian!",
    "category": "Function",
    "text": "ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x))\n\nCompute H(f) (i.e. J(∇(f))) evaluated at x and store the result(s) in result, assuming f is called as f(x).\n\nThis method assumes that isa(f(x), Real).\n\n\n\nForwardDiff.hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, result, x))\n\nExactly like ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig), but because isa(result, DiffResult), cfg is constructed as HessianConfig(f, result, x) instead of HessianConfig(f, x).\n\n\n\n"
},

{
    "location": "user/api.html#Hessians-of-f(x::AbstractArray)::Real-1",
    "page": "Differentiation API",
    "title": "Hessians of f(x::AbstractArray)::Real",
    "category": "section",
    "text": "ForwardDiff.hessian\nForwardDiff.hessian!"
},

{
    "location": "user/api.html#ForwardDiff.DerivativeConfig",
    "page": "Differentiation API",
    "title": "ForwardDiff.DerivativeConfig",
    "category": "Type",
    "text": "ForwardDiff.DerivativeConfig(f!, y::AbstractArray, x::AbstractArray)\n\nReturn a DerivativeConfig instance based on the type of f!, and the types/shapes of the output vector y and the input vector x.\n\nThe returned DerivativeConfig instance contains all the work buffers required by ForwardDiff.derivative and ForwardDiff.derivative! when the target function takes the form f!(y, x).\n\nThis constructor does not store/modify y or x.\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.GradientConfig",
    "page": "Differentiation API",
    "title": "ForwardDiff.GradientConfig",
    "category": "Type",
    "text": "ForwardDiff.GradientConfig(f, x::AbstractArray, chunk::Chunk = Chunk(x))\n\nReturn a GradientConfig instance based on the type of f and type/shape of the input vector x.\n\nThe returned GradientConfig instance contains all the work buffers required by ForwardDiff.gradient and ForwardDiff.gradient!.\n\nThis constructor does not store/modify x.\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.JacobianConfig",
    "page": "Differentiation API",
    "title": "ForwardDiff.JacobianConfig",
    "category": "Type",
    "text": "ForwardDiff.JacobianConfig(f, x::AbstractArray, chunk::Chunk = Chunk(x))\n\nReturn a JacobianConfig instance based on the type of f and type/shape of the input vector x.\n\nThe returned JacobianConfig instance contains all the work buffers required by ForwardDiff.jacobian and ForwardDiff.jacobian! when the target function takes the form f(x).\n\nThis constructor does not store/modify x.\n\n\n\nForwardDiff.JacobianConfig(f!, y::AbstractArray, x::AbstractArray, chunk::Chunk = Chunk(x))\n\nReturn a JacobianConfig instance based on the type of f!, and the types/shapes of the output vector y and the input vector x.\n\nThe returned JacobianConfig instance contains all the work buffers required by ForwardDiff.jacobian and ForwardDiff.jacobian! when the target function takes the form f!(y, x).\n\nThis constructor does not store/modify y or x.\n\n\n\n"
},

{
    "location": "user/api.html#ForwardDiff.HessianConfig",
    "page": "Differentiation API",
    "title": "ForwardDiff.HessianConfig",
    "category": "Type",
    "text": "ForwardDiff.HessianConfig(f, x::AbstractArray, chunk::Chunk = Chunk(x))\n\nReturn a HessianConfig instance based on the type of f and type/shape of the input vector x.\n\nThe returned HessianConfig instance contains all the work buffers required by ForwardDiff.hessian and ForwardDiff.hessian!. For the latter, the buffers are configured for the case where the result argument is an AbstractArray. If it is a DiffResult, the HessianConfig should instead be constructed via ForwardDiff.HessianConfig(f, result, x, chunk).\n\nThis constructor does not store/modify x.\n\n\n\nForwardDiff.HessianConfig(f, result::DiffResult, x::AbstractArray, chunk::Chunk = Chunk(x))\n\nReturn a HessianConfig instance based on the type of f, types/storage in result, and type/shape of the input vector x.\n\nThe returned HessianConfig instance contains all the work buffers required by ForwardDiff.hessian! for the case where the result argument is an DiffResult.\n\nThis constructor does not store/modify x.\n\n\n\n"
},

{
    "location": "user/api.html#Preallocating/Configuring-Work-Buffers-1",
    "page": "Differentiation API",
    "title": "Preallocating/Configuring Work Buffers",
    "category": "section",
    "text": "For the sake of convenience and performance, all \"extra\" information used by ForwardDiff's API methods is bundled up in the ForwardDiff.AbstractConfig family of types. These types allow the user to easily feed several different parameters to ForwardDiff's API methods, such as chunk size, work buffers, and perturbation seed configurations.ForwardDiff's basic API methods will allocate these types automatically by default, but you can drastically reduce memory usage if you preallocate them yourself.Note that for all constructors below, the chunk size N may be explicitly provided, or omitted, in which case ForwardDiff will automatically select a chunk size for you. However, it is highly recommended to specify the chunk size manually when possible (see Configuring Chunk Size).Note also that configurations constructed for a specific function f cannot be reused to differentiate other functions (though can be reused to differentiate f at different values). To construct a configuration which can be reused to differentiate any function, you can pass nothing as the function argument. While this is more flexible, it decreases ForwardDiff's ability to catch and prevent perturbation confusion.ForwardDiff.DerivativeConfig\nForwardDiff.GradientConfig\nForwardDiff.JacobianConfig\nForwardDiff.HessianConfig"
},

{
    "location": "user/advanced.html#",
    "page": "Advanced Usage Guide",
    "title": "Advanced Usage Guide",
    "category": "page",
    "text": ""
},

{
    "location": "user/advanced.html#Advanced-Usage-Guide-1",
    "page": "Advanced Usage Guide",
    "title": "Advanced Usage Guide",
    "category": "section",
    "text": "This document describes several techniques and features that can be used in conjunction ForwardDiff's basic API in order to fine-tune calculations and increase performance."
},

{
    "location": "user/advanced.html#Retrieving-Lower-Order-Results-1",
    "page": "Advanced Usage Guide",
    "title": "Retrieving Lower-Order Results",
    "category": "section",
    "text": "Let's say you want to calculate the value, gradient, and Hessian of some function f at an input x. You could execute f(x), ForwardDiff.gradient(f, x) and ForwardDiff.hessian(f, x), but that would be a horribly redundant way to accomplish this task!In the course of calculating higher-order derivatives, ForwardDiff ends up calculating all the lower-order derivatives and primal value f(x). To retrieve these results in one fell swoop, you can utilize the DiffResult API provided by the DiffBase package. To learn how to use this functionality, please consult the relevant DiffBase documentation.Note that running using ForwardDiff will automatically bring the DiffBase module into scope, and that all mutating ForwardDiff API methods support the DiffResult API. In other words, API methods of the form ForwardDiff.method!(out, args...) will work appropriately if isa(out, DiffResult)."
},

{
    "location": "user/advanced.html#Configuring-Chunk-Size-1",
    "page": "Advanced Usage Guide",
    "title": "Configuring Chunk Size",
    "category": "section",
    "text": "ForwardDiff performs partial derivative evaluation on one \"chunk\" of the input vector at a time. Each differentiation of a chunk requires a call to the target function as well as additional memory proportional to the square of the chunk's size. Thus, a smaller chunk size makes better use of memory bandwidth at the cost of more calls to the target function, while a larger chunk size reduces calls to the target function at the cost of more memory bandwidth.For example:julia> using ForwardDiff: GradientConfig, Chunk, gradient!\n\n# let's use a Rosenbrock function as our target function\njulia> function rosenbrock(x)\n           a = one(eltype(x))\n           b = 100 * a\n           result = zero(eltype(x))\n           for i in 1:length(x)-1\n               result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2\n           end\n           return result\n       end\nrosenbrock (generic function with 1 method)\n\n# input vector\njulia> x = rand(10000);\n\n# output buffer\njulia> out = similar(x);\n\n# construct GradientConfig with chunk size of 1\njulia> cfg1 = GradientConfig(rosenbrock, x, Chunk{1}());\n\n# construct GradientConfig with chunk size of 4\njulia> cfg4 = GradientConfig(rosenbrock, x, Chunk{4}());\n\n# construct GradientConfig with chunk size of 10\njulia> cfg10 = GradientConfig(rosenbrock, x, Chunk{10}());\n\n# (input length of 10000) / (chunk size of 1) = (10000 1-element chunks)\njulia> @time gradient!(out, rosenbrock, x, cfg1);\n  0.775139 seconds (4 allocations: 160 bytes)\n\n# (input length of 10000) / (chunk size of 4) = (2500 4-element chunks)\njulia> @time gradient!(out, rosenbrock, x, cfg4);\n  0.386459 seconds (4 allocations: 160 bytes)\n\n# (input length of 10000) / (chunk size of 10) = (1000 10-element chunks)\njulia> @time gradient!(out, rosenbrock, x, cfg10);\n  0.282529 seconds (4 allocations: 160 bytes)If you do not explicity provide a chunk size, ForwardDiff will try to guess one for you based on your input vector:# The GradientConfig constructor will automatically select a\n# chunk size in one is not explicitly provided\njulia> cfg = ForwardDiff.GradientConfig(rosenbrock, x);\n\njulia> @time ForwardDiff.gradient!(out, rosenbrock, x, cfg);\n  0.281853 seconds (4 allocations: 160 bytes)If your input dimension is constant across calls, you should explicitly select a chunk size rather than relying on ForwardDiff's heuristic. There are two reasons for this. The first is that ForwardDiff's heuristic depends only on the input dimension, whereas in reality the optimal chunk size will also depend on the target function. The second is that ForwardDiff's heuristic is inherently type-unstable, which can cause the entire call to be type-unstable.If your input dimension is a runtime variable, you can rely on ForwardDiff's heuristic without sacrificing type stability by manually asserting the output type, or - even better - by using the in-place API functions:# will be type-stable since you're asserting the output type\nForwardDiff.gradient(rosenbrock, x)::Vector{Float64}\n\n# will be type-stable since `out` is returned, and Julia knows the type of `out`\nForwardDiff.gradient!(out, rosenbrock, x)One final question remains: How should you select a chunk size? The answer is essentially \"perform your own benchmarks and see what works best for your use case.\" As stated before, the optimal chunk size is heavily dependent on the target function and length of the input vector.Note that it is usually best to pick a chunk size which divides evenly into the input dimension. Otherwise, ForwardDiff has to construct and utilize an extra \"remainder\" chunk to complete the calculation."
},

{
    "location": "user/advanced.html#Fixing-NaN/Inf-Issues-1",
    "page": "Advanced Usage Guide",
    "title": "Fixing NaN/Inf Issues",
    "category": "section",
    "text": "ForwardDiff's default behavior is to return NaN for undefined derivatives (or otherwise mirror the behavior of the function in Base, if it would return an error). This is usually the correct thing to do, but in some cases can erroneously \"poison\" values which aren't sensitive to the input and thus cause ForwardDiff to incorrectly return NaN or Inf derivatives. For example:# the dual number's perturbation component is zero, so this\n# variable should not propagate derivative information\njulia> log(ForwardDiff.Dual{:tag}(0.0, 0.0))\nDual{:tag}(-Inf,NaN) # oops, this NaN should be 0.0Here, ForwardDiff computes the derivative of log(0.0) as NaN and then propagates this derivative by multiplying it by the perturbation component. Usually, ForwardDiff can rely on the identity x * 0.0 == 0.0 to prevent the derivatives from propagating when the perturbation component is 0.0. However, this identity doesn't hold if isnan(y) or isinf(y), in which case a NaN derivative will be propagated instead.It is possible to fix this behavior by checking that the perturbation component is zero before attempting to propagate derivative information, but this check can noticeably decrease performance (~5%-10% on our benchmarks).In order to preserve performance in the majority of use cases, ForwardDiff disables this check by default. If your code is affected by this NaN behvaior, you can enable ForwardDiff's NaN-safe mode by setting NANSAFE_MODE_ENABLED to true in ForwardDiff's source. This constant is located in src/ForwardDiff.jl in the package's directory.In the future, we plan on allowing users and downstream library authors to dynamically enable NaN-safe mode via theAbstractConfig` API."
},

{
    "location": "user/advanced.html#Hessian-of-a-vector-valued-function-1",
    "page": "Advanced Usage Guide",
    "title": "Hessian of a vector-valued function",
    "category": "section",
    "text": "While ForwardDiff does not have a built-in function for taking Hessians of vector-valued functions, you can easily compose calls to ForwardDiff.jacobian to accomplish this. For example:julia> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(cumprod, x), [1,2,3])\n9×3 Array{Int64,2}:\n 0  0  0\n 0  1  0\n 0  3  2\n 0  0  0\n 1  0  0\n 3  0  1\n 0  0  0\n 0  0  0\n 2  1  0Since this functionality is composed from ForwardDiff's existing API rather than built into it, you're free to construct a vector_hessian function which suits your needs. For example, if you require the shape of the output to be a tensor rather than a block matrix, you can do so with a reshape (note that reshape does not copy data, so it's not an expensive operation):julia> function vector_hessian(f, x)\n       n = length(x)\n       out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)\n       return reshape(out, n, n, n)\n   end\nvector_hessian (generic function with 1 method)\n\njulia> vector_hessian(cumprod, [1, 2, 3])\n3×3×3 Array{Int64,3}:\n[:, :, 1] =\n 0  0  0\n 0  1  0\n 0  3  2\n\n[:, :, 2] =\n 0  0  0\n 1  0  0\n 3  0  1\n\n[:, :, 3] =\n 0  0  0\n 0  0  0\n 2  1  0Likewise, you could write a version of vector_hessian which supports functions of the form f!(y, x), or perhaps an in-place Jacobian with ForwardDiff.jacobian!."
},

{
    "location": "user/advanced.html#SIMD-Vectorization-1",
    "page": "Advanced Usage Guide",
    "title": "SIMD Vectorization",
    "category": "section",
    "text": "Many operations on ForwardDiff's dual numbers are amenable to SIMD vectorization. For some ForwardDiff benchmarks, we've seen SIMD vectorization yield speedups of almost 3x.To enable SIMD optimizations, start your Julia process with the -O3 flag. This flag enables LLVM's SLPVectorizerPass during compilation, which attempts to automatically insert SIMD instructions where possible for certain arithmetic operations.Here's an example of LLVM bitcode generated for an addition of two Dual numbers without SIMD instructions (i.e. not starting Julia with -O3):julia> using ForwardDiff: Dual\n\njulia> a = Dual(1., 2., 3., 4.)\nDual{Void}(1.0,2.0,3.0,4.0)\n\njulia> b = Dual(5., 6., 7., 8.)\nDual{Void}(5.0,6.0,7.0,8.0)\n\njulia> @code_llvm a + b\n\ndefine void @\"julia_+_70852\"(%Dual* noalias sret, %Dual*, %Dual*) #0 {\ntop:\n  %3 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 0\n  %4 = load double, double* %3, align 8\n  %5 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 0\n  %6 = load double, double* %5, align 8\n  %7 = fadd double %4, %6\n  %8 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 1\n  %9 = load double, double* %8, align 8\n  %10 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 1\n  %11 = load double, double* %10, align 8\n  %12 = fadd double %9, %11\n  %13 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 1, i32 0, i64 2\n  %14 = load double, double* %13, align 8\n  %15 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 1, i32 0, i64 2\n  %16 = load double, double* %15, align 8\n  %17 = fadd double %14, %16\n  %18 = getelementptr inbounds %Dual, %Dual* %1, i64 0, i32 0\n  %19 = load double, double* %18, align 8\n  %20 = getelementptr inbounds %Dual, %Dual* %2, i64 0, i32 0\n  %21 = load double, double* %20, align 8\n  %22 = fadd double %19, %21\n  %23 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 0\n  store double %22, double* %23, align 8\n  %24 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 0\n  store double %7, double* %24, align 8\n  %25 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 1\n  store double %12, double* %25, align 8\n  %26 = getelementptr inbounds %Dual, %Dual* %0, i64 0, i32 1, i32 0, i64 2\n  store double %17, double* %26, align 8\n  ret void\n}If we start up Julia with -O3 instead, the call to @code_llvm will show that LLVM can SIMD-vectorize the addition:julia> @code_llvm a + b\n\ndefine void @\"julia_+_70842\"(%Dual* noalias sret, %Dual*, %Dual*) #0 {\ntop:\n  %3 = bitcast %Dual* %1 to <4 x double>*            # cast the Dual to a SIMD-able LLVM vector\n  %4 = load <4 x double>, <4 x double>* %3, align 8\n  %5 = bitcast %Dual* %2 to <4 x double>*\n  %6 = load <4 x double>, <4 x double>* %5, align 8\n  %7 = fadd <4 x double> %4, %6                      # SIMD add\n  %8 = bitcast %Dual* %0 to <4 x double>*\n  store <4 x double> %7, <4 x double>* %8, align 8\n  ret void\n}Note that whether or not SIMD instructions can actually be used will depend on your machine and Julia build. For example, pre-built Julia binaries might not emit vectorized LLVM bitcode. To overcome this specific issue, you can locally rebuild Julia's system image."
},

{
    "location": "user/upgrade.html#",
    "page": "Upgrading from Older Versions",
    "title": "Upgrading from Older Versions",
    "category": "page",
    "text": ""
},

{
    "location": "user/upgrade.html#Upgrading-from-Older-Versions-1",
    "page": "Upgrading from Older Versions",
    "title": "Upgrading from Older Versions",
    "category": "section",
    "text": "Each new minor release of ForwardDiff may introduce changes in the API or behavior. Here, we'll provide some examples that highlight the key differences to help you upgrade from older versions of ForwardDiff."
},

{
    "location": "user/upgrade.html#Unexported-API-Functions-1",
    "page": "Upgrading from Older Versions",
    "title": "Unexported API Functions",
    "category": "section",
    "text": "In order to avoid namespace conflicts with other packages, ForwardDiff's  Differentiation API is no longer exported by default. Thus, you must now fully qualify the functions to  reference them:# ForwardDiff v0.1\nusing ForwardDiff\nhessian(f, x)\n\n# ForwardDiff v0.2 & above\nusing ForwardDiff\nForwardDiff.hessian(f, x)"
},

{
    "location": "user/upgrade.html#Setting-Chunk-Size-1",
    "page": "Upgrading from Older Versions",
    "title": "Setting Chunk Size",
    "category": "section",
    "text": "# ForwardDiff v0.1\nForwardDiff.gradient(f, x; chunk_size = 10)\n\n# ForwardDiff v0.2\nForwardDiff.gradient(f, x, Chunk{10}())\n\n# ForwardDiff v0.3 & v0.4\nForwardDiff.gradient(f, x, ForwardDiff.GradientConfig{10}(x))\n\n# ForwardDiff v0.5 & above\nForwardDiff.gradient(f, x, ForwardDiff.GradientConfig(f, x ForwardDiff.Chunk{N}()))"
},

{
    "location": "user/upgrade.html#Enabling-Multithreading-1",
    "page": "Upgrading from Older Versions",
    "title": "Enabling Multithreading",
    "category": "section",
    "text": "# ForwardDiff v0.1 & v0.2\nForwardDiff.gradient(f, x; multithread = true)\n\n# ForwardDiff v0.3 & v0.4\nForwardDiff.gradient(f, x, ForwardDiff.MultithreadConfig(ForwardDiff.GradientConfig(x)))\n\n# ForwardDiff v0.5 & above\nerror(\"ForwardDiff no longer supports internal multithreading.\")"
},

{
    "location": "user/upgrade.html#Retrieving-Lower-Order-Results-1",
    "page": "Upgrading from Older Versions",
    "title": "Retrieving Lower-Order Results",
    "category": "section",
    "text": "# ForwardDiff v0.1\nanswer, results = ForwardDiff.hessian(f, x, AllResults)\nv = ForwardDiff.value(results)\ng = ForwardDiff.gradient(results)\nh = ForwardDiff.hessian(results) # == answer\n\n# ForwardDiff v0.2\nout = HessianResult(x)\nForwardDiff.hessian!(out, f, x)\nv = ForwardDiff.value(out)\ng = ForwardDiff.gradient(out)\nh = ForwardDiff.hessian(out)\n\n# ForwardDiff v0.3 & above\nusing DiffBase\nout = DiffBase.HessianResult(x)\nForwardDiff.hessian!(out, f, x)\nv = DiffBase.value(out)\ng = DiffBase.gradient(out)\nh = DiffBase.hessian(out)"
},

{
    "location": "user/upgrade.html#Higher-Order-Differentiation-1",
    "page": "Upgrading from Older Versions",
    "title": "Higher-Order Differentiation",
    "category": "section",
    "text": "In order to maintain feature parity between all API functions, ForwardDiff no longer provides the tensor function. Instead, users can take higher-order/higher-dimensional derivatives by composing existing API functions. For example, here's how to reimplement tensor:# ForwardDiff v0.1\nForwardDiff.tensor(f, x)\n\n# ForwardDiff v0.2 & above\nfunction tensor(f, x)\n    n = length(x)\n    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)\n    return reshape(out, n, n, n)\nend\n\ntensor(f, x)"
},

{
    "location": "user/upgrade.html#Creating-Differentiation-Functions-1",
    "page": "Upgrading from Older Versions",
    "title": "Creating Differentiation Functions",
    "category": "section",
    "text": "Since v0.2, ForwardDiff no longer supports automatic generation of differentiation functions. Instead, users explicitly define their own functions using ForwardDiff's API. This leads to clearer code, less \"magic\", and more flexibility.# ForwardDiff v0.1\ndf = ForwardDiff.derivative(f)\n\n# ForwardDiff v0.2 & above\ndf = x -> ForwardDiff.derivative(f, x)# ForwardDiff v0.1\n# in-place gradient function of f\ngf! = ForwardDiff.gradient(f, mutates = true)\n\n# ForwardDiff v0.2 & above\ngf! = (out, x) -> ForwardDiff.gradient!(out, f, x)# ForwardDiff v0.1\n# in-place Jacobian function of f!(y, x):\njf! = ForwardDiff.jacobian(f!, mutates = true, output_length = length(y))\n\n# ForwardDiff v0.2 & above\njf! = (out, y, x) -> ForwardDiff.jacobian!(out, f!, y, x)"
},

{
    "location": "dev/how_it_works.html#",
    "page": "How ForwardDiff Works",
    "title": "How ForwardDiff Works",
    "category": "page",
    "text": ""
},

{
    "location": "dev/how_it_works.html#How-ForwardDiff-Works-1",
    "page": "How ForwardDiff Works",
    "title": "How ForwardDiff Works",
    "category": "section",
    "text": "ForwardDiff is an implementation of forward mode automatic differentiation (AD) in Julia. There are two key components of this implementation: the Dual type, and the API."
},

{
    "location": "dev/how_it_works.html#Dual-Number-Implementation-1",
    "page": "How ForwardDiff Works",
    "title": "Dual Number Implementation",
    "category": "section",
    "text": "Partial derivatives are stored in the Partials type:struct Partials{N,V} <: AbstractVector{V}\n    values::NTuple{N,V}\nendThe Partials type is used to implement the Dual type:struct Dual{T,V<:Real,N} <: Real\n    value::V\n    partials::Partials{N,V}\nendThis type represents an N-dimensional dual number coupled with a tag parameter T in order to prevent perturbation confusion. This dual number type is implemented to have the following mathematical behavior:f(a + sum_i=1^N b_i epsilon_i) = f(a) + f(a) sum_i=1^N b_i epsilon_iwhere the a component is stored in the value field and the b components are stored in the partials field. This property of dual numbers is the central feature that allows ForwardDiff to take derivatives.In order to implement the above property, elementary numerical functions on a Dual number are overloaded to evaluate both the original function, and evaluate the derivative of the function, propogating the derivative via multiplication. For example, Base.sin can be overloaded on Dual like so:Base.sin(d::Dual{T}) where {T} = Dual{T}(sin(value(d)), cos(value(d)) * partials(d))If we assume that a general function f is composed of entirely of these elementary functions, then the chain rule enables our derivatives to compose as well. Thus, by overloading a plethora of elementary functions, we can differentiate generic functions composed of them by passing in a Dual number and looking at the output.We won't discuss higher-order differentiation in detail, but the reader is encouraged to learn about hyper-dual numbers, which extend dual numbers to higher orders by introducing extra epsilon terms that can cross-multiply. ForwardDiff's Dual number implementation naturally supports hyper-dual numbers without additional code by allowing instances of the Dual type to nest within each other. For example, a second-order hyper-dual number has the type Dual{T,Dual{S,V,M},N}, a third-order hyper-dual number has the type Dual{T,Dual{S,Dual{R,V,K},M},N}, and so on."
},

{
    "location": "dev/how_it_works.html#ForwardDiff's-API-1",
    "page": "How ForwardDiff Works",
    "title": "ForwardDiff's API",
    "category": "section",
    "text": "The second component provided by this package is the API, which abstracts away the number types and makes it easy to execute familiar calculations like gradients and Hessians. This way, users don't have to understand Dual numbers in order to make use of the package.The job of the API functions is to performantly seed input values with Dual numbers, pass the seeded value into the target function, and extract the derivative information from the result. For example, to calculate the partial derivatives for the gradient of a function f at an input vector vecx, we would do the following:vecx = beginbmatrix\n               x_1 \n               vdots \n               x_i \n               vdots \n               x_N\n           endbmatrix\nto\nvecx_epsilon = beginbmatrix\n                         x_1 + epsilon_1 \n                         vdots \n                         x_i + epsilon_i \n                         vdots \n                         x_N + epsilon_N\n                     endbmatrix\nto\nf(vecx_epsilon) = f(vecx) + sum_i=1^N fracdelta f(vecx)delta x_i epsilon_iIn reality, ForwardDiff does this calculation in chunks of the input vector (see Configuring Chunk Size for details). To provide a simple example of this, let's examine the case where the input vector size is 4 and the chunk size is 2. It then takes two calls to f to evaluate the gradient:vecx = beginbmatrix\n               x_1 \n               x_2 \n               x_3 \n               x_4\n           endbmatrix\n\nvecx_epsilon = beginbmatrix\n                        x_1 + epsilon_1 \n                        x_2 + epsilon_2 \n                        x_3 \n                        x_4\n                     endbmatrix\nto\nf(vecx_epsilon) = f(vecx) + fracdelta f(vecx)delta x_1 epsilon_1 + fracdelta f(vecx)delta x_2 epsilon_2\n\nvecx_epsilon = beginbmatrix\n                        x_1 \n                        x_2 \n                        x_3 + epsilon_1 \n                        x_4 + epsilon_2\n                     endbmatrix\nto\nf(vecx_epsilon) = f(vecx) + fracdelta f(vecx)delta x_3 epsilon_1 + fracdelta f(vecx)delta x_4 epsilon_2This seeding process is similar for Jacobians, so we won't rehash it here."
},

{
    "location": "dev/contributing.html#",
    "page": "How to Contribute",
    "title": "How to Contribute",
    "category": "page",
    "text": ""
},

{
    "location": "dev/contributing.html#How-to-Contribute-1",
    "page": "How to Contribute",
    "title": "How to Contribute",
    "category": "section",
    "text": "There are a few fairly easy ways for newcomers to substantially improve ForwardDiff, and they all revolve around writing functions for Dual numbers. This section provides brief tutorials on how to make these contributions.If you're new GitHub, here's an outline of the workflow you should use:Fork ForwardDiff\nMake a new branch on your fork, named after whatever changes you'll be making\nApply your code changes to the branch on your fork\nWhen you're done, submit a PR to ForwardDiff to merge your fork into ForwardDiff's master branch."
},

{
    "location": "dev/contributing.html#Adding-New-Derivative-Definitions-1",
    "page": "How to Contribute",
    "title": "Adding New Derivative Definitions",
    "category": "section",
    "text": "In general, new derivative implementations for Dual are automatically defined via simple symbolic rules. ForwardDiff accomplishes this by looping over the the function names listed in the RealInterface package, and for every function (and relevant arity), it attempts to generate a Dual definition by applying the symbolic rules provided by the DiffBase package. Conveniently, these auto-generated definitions are also automatically tested.Thus, in order to add a new derivative implementation for Dual, you should do the following:Make sure the name of the function is appropriately listed in the RealInterface package\nDefine the appropriate derivative rule(s) in DiffBase\nCheck that calling the function on Dual instances delivers the desired result.Depending on the arity of your function and its category in RealInterface, ForwardDiff's auto-definition mechanism might need to be expanded to include it. If this is the case, ForwardDiff's maintainers can help you out."
},

]}
