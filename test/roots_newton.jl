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


function fp_dual(x)
    res = f(dual(x, 1.))
    return res.du
end
root2a = newton(f, fp_dual, 3.0)

# Source-code transformation

function f1(x)
    z = x    # to add a wrinkle
    exp(z) - sin(x)
end
autodiff_transform(f1, (Float64,)) 

autodiff_transform(f1, :f1d, (Float64,)) 

function fp_sct(x)
    res = f1(ad(x, 1.))
    return gradient(res)[1]
end

root3 = newton(f, fp_sct, 3.0)


N = 1000000
f1() = for i in 1:N r = fp_ad(3.0) end
f2() = for i in 1:N r = fp_sct(3.0) end
@time f1()
@time f2()


# Test function speed

N = 1000000
function f0()
    a = 3.
    for i in 1:N
        r = f(a)
    end
end
function f1()     # Dual numbers with operator overload
    a = Dual(3., 1.)
    for i in 1:N
        r = f(a)
    end
end
function f2()     # ADForward with operator overload
    a = ad(3., 1.)
    for i in 1:N
        r = f(a)
    end
end
function f3()     # Source transformation with dual numbers
    a = Dual(3., 1.)
    for i in 1:N
        r = f1(a)
    end
end

@time f0()   # base function           elapsed time: 0.151144737 seconds
@time f1()   # Dual numbers            elapsed time: 0.510655266 seconds
@time f2()   # ADForward               elapsed time: 1.903879642 seconds
@time f3()   # source transformation   elapsed time: 0.388411608 seconds
