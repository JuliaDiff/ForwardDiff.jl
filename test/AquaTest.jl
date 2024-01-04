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

    @test length(ambs) == 0
end

@testset "Aqua tests - remaining" begin
    Aqua.test_all(ForwardDiff; ambiguities = false, unbound_args = false)
end

nothing
