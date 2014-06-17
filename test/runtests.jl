tests = ["dual_fad", "GraDual", "FADHessian", "FADTensor"]

println("Running tests:")

for t in tests
    tfile = "$t.jl"
    println(" * $tfile")
    include(tfile)
end
