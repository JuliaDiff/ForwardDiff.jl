using NLopt

## Rosenbrock Banana function
function f(x::Vector, grad::Vector) 
    if length(grad) > 0
        grad[1] = -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1])
        grad[2] = 200 * (x[2] - x[1] * x[1]) 
    end
    return 100 * (x[2] - x[1] * x[1])^2 + (1 - x[1])^2 
end

function f1(x::Vector, grad::Vector = []) 
    return 100. * (x[2] - x[1] * x[1])^2 + (1. - x[1])^2 
end

x0 = [ -1.2, 1 ]

opt = Opt(:LD_LBFGS, 2)
min_objective!(opt, f)
xtol_rel!(opt,1e-8)

(minf,minx,ret) = optimize(opt, x0)



function f_ad(x::Vector, grad::Vector)
    res = f1(ad(x))
    if length(grad) > 0
        grad[:] = Gradient(res)
    end
    return Value(res)
end

opt2 = Opt(:LD_LBFGS, 2)
min_objective!(opt2, f_ad)
xtol_rel!(opt2,1e-8)

(minf2,minx2,ret2) = optimize(opt2, x0)
