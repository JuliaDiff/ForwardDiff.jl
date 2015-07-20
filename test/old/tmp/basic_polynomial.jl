# speed tests comparing the base function to various AutoDiff routines
# (Dual numbers, ADForward, and source transformation).


using AutoDiff

f1(x) = x^4 + x^3 + (x^2 + x) * x
f2(x) = x^4 + x^3 + (x^2 + x) * x

autodiff_transform(f2, (Float64,)) 

N = 1000000
function f0()
    a = 3.
    for i in 1:N
        r = f1(a)
    end
end
function f1()
    a = 3. + 1im
    for i in 1:N
        r = f1(a)
    end
end
function f2()
    a = Dual(3., 1.)
    for i in 1:N
        r = f1(a)
    end
end
function f3()
    a = ad(3., 1.)
    for i in 1:N
        r = f1(a)
    end
end
function f4()
    a = Dual(3., 1.)
    for i in 1:N
        r = f2(a)
    end
end

@time f0()   # base function           elapsed time: 0.110321383 seconds
@time f1()   # complex numbers         elapsed time: 0.255644874 seconds
@time f2()   # Dual numbers            elapsed time: 0.397590266 seconds
@time f3()   # ADForward               elapsed time: 12.348061681 seconds
@time f4()   # source transformation   elapsed time: 0.148293506 seconds
