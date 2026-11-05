using ForwardDiff
include("functions.jl")

function (@main)(args::Vector{String})::Cint
    n = parse(Int, args[1])
    foreach(a -> foreach(v -> println(Core.stdout, v), a), results(inputs(n)))
    return 0
end
