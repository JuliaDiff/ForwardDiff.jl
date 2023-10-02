module ComplexTest
using ForwardDiff, Test

@testset "complex dual" begin
    x = Dual(1., 2., 3.) + im*Dual(4.,5.,6.)
    @test value(x) == 1 + 4im
    @test partials(x,1) == 2 + 5im
    @test partials(x,2) == 3 + 6im
end
end