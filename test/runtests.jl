using ForwardDiff

function fuzz_R4_to_R(allowed_funcs)
    uni_funcs = rand(collect(allowed_funcs), 4)
    bi_funcs = rand([:*, :/, :+, :-, :^], 3)
    return quote
        $(bi_funcs[3])($(bi_funcs[2])($(bi_funcs[1])($(uni_funcs[1])(a), $(uni_funcs[2])(b)), $(uni_funcs[3])(c)), $(uni_funcs[4])(d))
    end
end

include("test_gradnum.jl")
include("test_hessnum.jl")
include("test_tensnum.jl")
include("test_derivatives.jl")
include("test_jacobians.jl")