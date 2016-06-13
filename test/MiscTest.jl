module MiscTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

############################
# higher-order derivatives #
############################

function tensor(f, x)
    n = length(x)
    out = ForwardDiff.jacobian(y -> ForwardDiff.hessian(f, y), x)
    return reshape(out, n, n, n)
end

test_tensor_output = reshape([240.0  -400.0     0.0;
                             -400.0     0.0     0.0;
                                0.0     0.0     0.0;
                             -400.0     0.0     0.0;
                                0.0   480.0  -400.0;
                                0.0  -400.0     0.0;
                                0.0     0.0     0.0;
                                0.0  -400.0     0.0;
                                0.0     0.0     0.0], 3, 3, 3)

test_approx_eps(tensor(rosenbrock, [0.1, 0.2, 0.3]), test_tensor_output)

end
