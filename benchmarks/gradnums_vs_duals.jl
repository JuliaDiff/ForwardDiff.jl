using ForwardDiff.GradientNumber
using DualNumbers

# cmp_codegen(f) should report fairly similar code for 
# Duals and GradientNumbers, possibly with a few extra 
# allocations due to the Partials layer
#
# cmp_times(f) should report similar times for Duals 
# and GradientNumbers (within a few percent difference)

function cmp_codegen(f)
    x = rand(3);
    dx = map(dual, x);
    ndx = map(i->GradientNumber(i, zero(i)), x);

    info("@code_llvm $(f)(::Vector{Dual{Float64}}):")
    @code_llvm f(dx)

    println()

    info("@code_llvm $(f)(::Vector{GradientNumber{1,Float64,Tuple{Float64}}}):")
    @code_llvm f(ndx)
end

function cmp_times(f, xlen, repeat=5)
    x = rand(xlen);
    dx = map(dual, x);
    ndx = map(i->GradientNumber(i, zero(i)), x);

    min_time_dual = Inf
    min_time_gradnum = Inf

    for i in 1:(repeat+1) # +1 for warm-up
        gc()
        this_time_dual = @elapsed f(dx)
        this_time_gradnum = @elapsed f(ndx)
        min_time_dual = min(this_time_dual, min_time_dual)
        min_time_gradnum = min(this_time_gradnum, min_time_gradnum)
    end

    println("Time for $(f)(::Vector{Dual{Float64}}): $min_time_dual seconds")
    println("Time for $(f)(::Vector{GradientNumber{1,Float64,Tuple{Float64}}}): $min_time_gradnum seconds")
end
