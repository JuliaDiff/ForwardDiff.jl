module PartialsTest

using Base.Test
using ForwardDiff
using ForwardDiff.Partials

samerng() = MersenneTwister(1)

for N in (0, 3), T in (Int, Float32, Float64)
    println("  ...testing Partials{$N,$T}")

    VALUES = ntuple(n -> rand(T), Val{N})
    PARTIALS = Partials{N,T}(VALUES)

    VALUES2 = ntuple(n -> rand(T), Val{N})
    PARTIALS2 = Partials{N,T}(VALUES2)

    ##############################
    # Utility/Accessor Functions #
    ##############################

    @test PARTIALS.values == VALUES

    @test ForwardDiff.valtype(PARTIALS) == T
    @test ForwardDiff.valtype(typeof(PARTIALS)) == T

    @test ForwardDiff.npartials(PARTIALS) == N
    @test ForwardDiff.npartials(typeof(PARTIALS)) == N

    @test length(PARTIALS) == N
    @test length(VALUES) == N

    for i in 1:N
        @test PARTIALS[i] == VALUES[i]
    end

    @test start(PARTIALS) == start(VALUES)
    @test N == 0 || (next(PARTIALS, start(PARTIALS)) == next(VALUES, start(VALUES)))
    @test done(PARTIALS, start(PARTIALS)) == done(VALUES, start(VALUES))

    i = 1
    for p in PARTIALS
        @test p == VALUES[i]
        i += 1
    end

    #####################
    # Generic Functions #
    #####################

    @test zero(PARTIALS) == zero(typeof(PARTIALS))
    @test zero(PARTIALS).values == map(zero, VALUES)

    @test one(PARTIALS) == one(typeof(PARTIALS))
    @test one(PARTIALS).values == map(one, VALUES)

    @test rand(samerng(), PARTIALS) == rand(samerng(), typeof(PARTIALS))

    @test ForwardDiff.iszero(PARTIALS) == (N == 0)
    @test ForwardDiff.iszero(zero(PARTIALS))

    @test PARTIALS == copy(PARTIALS)
    @test (PARTIALS == PARTIALS2) == (N == 0)
    @test isequal(PARTIALS, copy(PARTIALS))
    @test isequal(PARTIALS, PARTIALS2) == (N == 0)

    @test hash(PARTIALS) == hash(VALUES, ForwardDiff.PARTIALS_HASH)
    @test hash(PARTIALS) == hash(copy(PARTIALS))
    @test hash(PARTIALS, hash(1)) == hash(copy(PARTIALS), hash(1))
    @test hash(PARTIALS, hash(1)) == hash(copy(PARTIALS), hash(1))

    const TMPIO = IOBuffer()
    write(TMPIO, PARTIALS)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(PARTIALS)) == PARTIALS
    seekstart(TMPIO)
    write(TMPIO, PARTIALS2)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(PARTIALS2)) == PARTIALS2
    close(TMPIO)

    ########################
    # Conversion/Promotion #
    ########################

    const WIDE_T = widen(T)
    const WIDE_PARTIALS = convert(Partials{N,WIDE_T}, PARTIALS)

    @test typeof(WIDE_PARTIALS) == Partials{N,WIDE_T}
    @test WIDE_PARTIALS == PARTIALS
    @test convert(Partials{N,T}, PARTIALS) === PARTIALS
    @test promote_type(Partials{N,T}, Partials{N,T}) == Partials{N,T}
    @test promote_type(Partials{N,T}, Partials{N,WIDE_T}) == Partials{N,WIDE_T}

    ########################
    # Arithmetic Functions #
    ########################

    @test (PARTIALS + PARTIALS).values == map(v -> v + v, VALUES)
    @test (PARTIALS - PARTIALS).values == map(v -> v - v, VALUES)
    @test getfield(-(PARTIALS), :values) == map(-, VALUES)

    const X = rand()
    const Y = rand()

    @test X * PARTIALS == PARTIALS * X
    @test (X * PARTIALS).values == map(v -> X * v, VALUES)
    @test (PARTIALS / X).values == map(v -> v / X, VALUES)

    if N > 0
        @test ForwardDiff._mul_partials(PARTIALS, PARTIALS2, X, Y).values == map((a, b) -> (X * a) + (Y * b), VALUES, VALUES2)
        @test ForwardDiff._div_partials(PARTIALS, PARTIALS2, X, Y) == ForwardDiff._mul_partials(PARTIALS, PARTIALS2, inv(Y), -X/(Y^2))
    end
end

end # module
