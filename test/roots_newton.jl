using AutoDiff
using Roots

f(x) = exp(x) - sin(x)
fp(x) = exp(x) - cos(x)

root = newton(f, fp, 3.0)
println(f(root))


function fp_ad(x)
    res = f(ad(x, 1.))
    return gradient(res)[1]
end
root2 = newton(f, fp_ad, 3.0)


# Source-code transformation

function f1(x)
    z = x    # to add a wrinkle
    exp(z) - sin(x)
end
autodiff_transform(f1, (Float64,)) 

function fp_sct(x)
    res = f1(ad(x, 1.))
    return gradient(res)[1]
end

root3 = newton(f, fp_sct, 3.0)
