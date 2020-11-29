module PartialsTest

using Test
using Random
using ForwardDiff
using ForwardDiff: Partials

samerng() = MersenneTwister(1)

for N in (0, 3), T in (Int, Float32, Float64)
    println("  ...testing Partials{$N,$T}")

    VALUES = (rand(T,N)...,)
    PARTIALS = Partials{N,T}(VALUES)

    VALUES2 = (rand(T,N)...,)
    PARTIALS2 = Partials{N,T}(VALUES2)

    ##############################
    # Utility/Accessor Functions #
    ##############################

    @test PARTIALS.values == VALUES

    @test ForwardDiff.valtype(PARTIALS) == T
    @test ForwardDiff.valtype(typeof(PARTIALS)) == T

    @test ForwardDiff.npartials(PARTIALS) == N
    @test ForwardDiff.npartials(typeof(PARTIALS)) == N

    @test ndims(PARTIALS) == ndims(PARTIALS2) == 1
    @test length(PARTIALS) == N
    @test length(VALUES) == N

    for i in 1:N
        @test PARTIALS[i] == VALUES[i]
    end

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

    TMPIO = IOBuffer()
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

    WIDE_T = widen(T)
    WIDE_PARTIALS = convert(Partials{N,WIDE_T}, PARTIALS)

    @test typeof(WIDE_PARTIALS) == Partials{N,WIDE_T}
    @test WIDE_PARTIALS == PARTIALS
    @test convert(Partials{N,T}, PARTIALS) === PARTIALS
    @test promote_type(Partials{N,T}, Partials{N,T}) == Partials{N,T}
    @test promote_type(Partials{N,T}, Partials{N,WIDE_T}) == Partials{N,WIDE_T}

    ########################
    # Arithmetic Functions #
    ########################

    ZERO_PARTIALS = Partials{0,T}(())

    @test (PARTIALS + PARTIALS).values == map(v -> v + v, VALUES)
    @test (PARTIALS + ZERO_PARTIALS) === PARTIALS
    @test (ZERO_PARTIALS + PARTIALS) === PARTIALS

    @test (PARTIALS - PARTIALS).values == map(v -> v - v, VALUES)
    @test (PARTIALS - ZERO_PARTIALS) === PARTIALS
    @test (ZERO_PARTIALS - PARTIALS) === -PARTIALS
    @test getfield(-(PARTIALS), :values) == map(-, VALUES)

    X = rand()
    Y = rand()

    @test X * PARTIALS == PARTIALS * X
    @test (X * PARTIALS).values == map(v -> X * v, VALUES)
    @test (PARTIALS / X).values == map(v -> v / X, VALUES)

    if N > 0
        @test ForwardDiff._div_partials(PARTIALS, PARTIALS2, X, Y) == ForwardDiff._mul_partials(PARTIALS, PARTIALS2, inv(Y), -X/(Y^2))
        @test ForwardDiff._mul_partials(PARTIALS, PARTIALS2, X, Y).values == map((a, b) -> (X * a) + (Y * b), VALUES, VALUES2)
        @test ForwardDiff._mul_partials(ZERO_PARTIALS, PARTIALS, X, Y) == Y * PARTIALS
        @test ForwardDiff._mul_partials(PARTIALS, ZERO_PARTIALS, X, Y) == X * PARTIALS

        if ForwardDiff.NANSAFE_MODE_ENABLED
            ZEROS = Partials((fill(zero(T), N)...,))

            @test (NaN * ZEROS).values == ZEROS.values
            @test (Inf * ZEROS).values == ZEROS.values
            @test (ZEROS / 0).values == ZEROS.values

            @test ForwardDiff._mul_partials(ZEROS, ZEROS, X, NaN).values == ZEROS.values
            @test ForwardDiff._mul_partials(ZEROS, ZEROS, NaN, X).values == ZEROS.values
            @test ForwardDiff._mul_partials(ZEROS, ZEROS, X, Inf).values == ZEROS.values
            @test ForwardDiff._mul_partials(ZEROS, ZEROS, Inf, X).values == ZEROS.values
            @test ForwardDiff._mul_partials(ZEROS, ZEROS, Inf, NaN).values == ZEROS.values
            @test ForwardDiff._mul_partials(ZEROS, ZEROS, NaN, Inf).values == ZEROS.values
        end
    end
end

io = IOBuffer()
show(io, MIME("text/plain"), Partials((1, 2, 3)))
str = String(take!(io))
@test str == "3-element $(ForwardDiff.Partials{3,Int}):\n 1\n 2\n 3"

end # module
