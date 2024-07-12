module AquaTest

using Aqua
using ForwardDiff
using Test

@testset "Aqua tests - remaining" begin
    # Test ambiguities separately without Base and Core
    # Ref: https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_all(ForwardDiff; ambiguities = false)
    Aqua.test_ambiguities(ForwardDiff)
end

end
