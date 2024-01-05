module AquaTest

using Test
using ForwardDiff
using Aqua

@testset "Aqua tests - unbound_args" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    ua = Aqua.detect_unbound_args_recursively(ForwardDiff)
    @test length(ua) == 6
end

@testset "Aqua tests - ambiguities" begin
    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ForwardDiff; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ForwardDiff", pkgdir(last(x).module)), ambs)

    ambs_dict = Dict()
    ambs_dict[(1, 6)] = 2
    ambs_dict[(1, 10)] = 1
    verkey(v) = (Int(VERSION.major), Int(VERSION.minor))

    if haskey(ambs_dict, verkey(VERSION))
        @test length(ambs) ≤ ambs_dict[verkey(VERSION)]
        # notify us when we fix one
        if length(ambs) < ambs_dict[verkey(VERSION)]
            @info "Ambiguities may have been fixed, please lower the limit."
            @info "     length(ambs) = $(length(ambs))"
        end
    end
end

@testset "Aqua tests - remaining" begin
    Aqua.test_all(ForwardDiff; ambiguities = false, unbound_args = false)
end

end
