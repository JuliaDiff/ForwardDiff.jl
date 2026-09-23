module PartialsTest

using Test
using Random
using ForwardDiff
using ForwardDiff: Partials

include(joinpath(dirname(@__FILE__), "Furlongs.jl"))

samerng() = MersenneTwister(1)

@testset "Partials{$N,$T}" for N in (0, 3), T in (Int, Float32, Float64)

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
    @test !(PARTIALS < copy(PARTIALS))
    @test PARTIALS <= copy(PARTIALS)
    @test !(PARTIALS > copy(PARTIALS))
    @test !isless(PARTIALS, copy(PARTIALS))
    for f in (<, <=, >, isless)
        @test f(PARTIALS, PARTIALS2) === f(VALUES, VALUES2)
    end
    NAN_PARTIALS = Partials{N,float(T)}(map(x -> oftype(float(x), NaN), VALUES))
    @test !(PARTIALS < NAN_PARTIALS)
    @test (PARTIALS <= NAN_PARTIALS) === (N == 0)
    @test !(PARTIALS > NAN_PARTIALS)
    @test isless(PARTIALS, NAN_PARTIALS) === (N > 0)
    @test !isless(NAN_PARTIALS, PARTIALS)

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
        # Only zero partials
        ALLZERO = Partials(ntuple(_ -> zero(T), N))
        # Mix of zero and non-zero partials
        FIRSTZERO = Partials(ntuple(i -> i == 1 ? zero(T) : rand(T), N))

        # The following properties should always be satisfied, regardless of whether NaN-safe mode is enabled or disabled
        # We use `isequal` for comparisons in the presence of `NaN`s
        for p1 in (PARTIALS, ALLZERO, FIRSTZERO), p2 in (PARTIALS2, ALLZERO, FIRSTZERO), v1 in (X, NaN, Inf), v2 in (Y, NaN, Inf)
            @test isequal(ForwardDiff._div_partials(p1, p2, v1, v2), ForwardDiff._mul_partials(p1, p2, inv(v2), -v1/(v2^2)))
            @test isequal(ForwardDiff._mul_partials(p1, p2, v1, v2), v1 * p1 + v2 * p2)
        end
        for v1 in (X, NaN, Inf), v2 in (Y, NaN, Inf)
            @test isequal(ForwardDiff._mul_partials(ZERO_PARTIALS, PARTIALS, v1, v2), v2 * PARTIALS)
            @test isequal(ForwardDiff._mul_partials(PARTIALS, ZERO_PARTIALS, v1, v2), v1 * PARTIALS)
        end

        if ForwardDiff.NANSAFE_MODE_ENABLED
            for f in ((p -> NaN * p), (p -> Inf * p), (p -> -Inf * p), (p -> p / 0), (p -> p / NaN), (p -> p / Inf), (p -> p / -Inf))
                # Only zero partials
                @test iszero(@inferred(f(ALLZERO)))

                # Mix of zero and non-zero partials
                z = @inferred(f(FIRSTZERO))
                for i in 1:N
                    if iszero(FIRSTZERO[i])
                        @test iszero(z[i])
                    else
                        @test isequal(z[i], f(FIRSTZERO[i]))
                    end
                end
            end
        end
    end

    @testset "non-standard numbers" begin # Will be fixed by changing single_seed to use oneunit rather than one
        @test_throws MethodError ForwardDiff.construct_seeds(ForwardDiff.Partials{3, Furlongs.Furlong{1, Float64}})
    end
end

io = IOBuffer()
show(io, MIME("text/plain"), Partials((1, 2, 3)))
str = String(take!(io))
@test str == "3-element $(ForwardDiff.Partials{3,Int}):\n 1\n 2\n 3"

end # module
