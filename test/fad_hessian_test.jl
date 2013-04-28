using AutoDiff

args = FADHessian([1., 2., 3.])
#f(x, y, z) = x*x*x+y*y*y+z*z*z+x*y*y+y*z*z+x*z*z
f(x, y, z) = x^3+y^3+z^3+x*y^2+y*z^2+x*z^2
w = f(args...)

println("f(x, y, z) = x^3+y^3+z^3+x*y^2+y*z^2+x*z^2\n")
println("f(1, 2, 3) = ", value(w), "\n")
println("Gradient of f:\n", grad(w), "\n")
println("Hessian H(f):\n", hessian(w))
