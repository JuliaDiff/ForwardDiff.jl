## Calculation of f(2) and f'(2) for function f:R->R given by f(x)=x^3

using AutoDiff

x = Dual(2., 1.)
f(x) = x^3
y = f(x)

println("f(x)=x^3")
println("f(2) = ", real(y))
println("f'(2) = ", imag(y), "\n")
show(y)
