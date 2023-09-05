using Test
using ForwardDiff
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    ua = Aqua.detect_unbound_args_recursively(ForwardDiff)
    @test length(ua) == 6

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(ForwardDiff; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("ForwardDiff", pkgdir(last(x).module)), ambs)

    @test length(ambs) == 0
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(ForwardDiff)
    Aqua.test_stale_deps(ForwardDiff)
    Aqua.test_deps_compat(ForwardDiff)
    Aqua.test_project_extras(ForwardDiff)
    Aqua.test_project_toml_formatting(ForwardDiff)
    Aqua.test_piracy(ForwardDiff)
end

nothing
