
using Roots

f(x) = exp(x) - sin(x)
fp(x) = exp(x) - cos(x)

root = newton(f, fp, 3.0)
println(f(root))


function fp_ad(x)
    res = f(ad(x, 1.))
    return Gradient(res)[1]
end
root2 = newton(f, fp_ad, 3.0)
