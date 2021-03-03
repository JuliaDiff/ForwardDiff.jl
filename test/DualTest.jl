module DualTest

using Test
using Random
using ForwardDiff
using ForwardDiff: Partials, Dual, value, partials

using NaNMath, SpecialFunctions
using DiffRules

import Calculus

struct TestTag end

samerng() = MersenneTwister(1)

# By lower-bounding the Int range at 2, we avoid cases where differentiating an
# exponentiation of an Int value would cause a DomainError due to reducing the
# exponent by one
intrand(V) = V == Int ? rand(2:10) : rand(V)

dual_isapprox(a, b) = isapprox(a, b)
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T,T3,T4}) where {T,T1,T2,T3,T4} = isapprox(value(a), value(b)) && isapprox(partials(a), partials(b))
dual_isapprox(a::Dual{T,T1,T2}, b::Dual{T3,T4,T5}) where {T,T1,T2,T3,T4,T5} = error("Tags don't match")

ForwardDiff.:≺(::Type{TestTag()}, ::Int) = true
ForwardDiff.:≺(::Int, ::Type{TestTag()}) = false

for N in (0,3), M in (0,4), V in (Int, Float32)
    println("  ...testing Dual{TestTag(),$V,$N} and Dual{TestTag(),Dual{TestTag(),$V,$M},$N}")

    PARTIALS = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL = intrand(V)
    FDNUM = Dual{TestTag()}(PRIMAL, PARTIALS)

    PARTIALS2 = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL2 = intrand(V)
    FDNUM2 = Dual{TestTag()}(PRIMAL2, PARTIALS2)

    PARTIALS3 = Partials{N,V}(ntuple(n -> intrand(V), N))
    PRIMAL3 = intrand(V)
    FDNUM3 = Dual{TestTag()}(PRIMAL3, PARTIALS3)

    M_PARTIALS = Partials{M,V}(ntuple(m -> intrand(V), M))
    NESTED_PARTIALS = convert(Partials{N,Dual{TestTag(),V,M}}, PARTIALS)
    NESTED_FDNUM = Dual{TestTag()}(Dual{TestTag()}(PRIMAL, M_PARTIALS), NESTED_PARTIALS)

    M_PARTIALS2 = Partials{M,V}(ntuple(m -> intrand(V), M))
    NESTED_PARTIALS2 = convert(Partials{N,Dual{TestTag(),V,M}}, PARTIALS2)
    NESTED_FDNUM2 = Dual{TestTag()}(Dual{TestTag()}(PRIMAL2, M_PARTIALS2), NESTED_PARTIALS2)

    ################
    # Constructors #
    ################

    @test Dual{TestTag()}(PRIMAL, PARTIALS...) === FDNUM
    @test Dual(PRIMAL, PARTIALS...) === Dual{Nothing}(PRIMAL, PARTIALS...)
    @test Dual(PRIMAL) === Dual{Nothing}(PRIMAL)

    @test typeof(Dual{TestTag()}(widen(V)(PRIMAL), PARTIALS)) === Dual{TestTag(),widen(V),N}
    @test typeof(Dual{TestTag()}(widen(V)(PRIMAL), PARTIALS.values)) === Dual{TestTag(),widen(V),N}
    @test typeof(Dual{TestTag()}(widen(V)(PRIMAL), PARTIALS...)) === Dual{TestTag(),widen(V),N}
    @test typeof(NESTED_FDNUM) == Dual{TestTag(),Dual{TestTag(),V,M},N}

    #############
    # Accessors #
    #############

    @test value(PRIMAL) == PRIMAL
    @test value(FDNUM) == PRIMAL
    @test value(NESTED_FDNUM) === Dual{TestTag()}(PRIMAL, M_PARTIALS)

    @test partials(PRIMAL) == Partials{0,V}(tuple())
    @test partials(FDNUM) == PARTIALS
    @test partials(NESTED_FDNUM) === NESTED_PARTIALS

    for i in 1:N
        @test partials(FDNUM, i) == PARTIALS[i]
        for j in 1:M
            @test partials(NESTED_FDNUM, i, j) == partials(NESTED_PARTIALS[i], j)
        end
    end

    @test ForwardDiff.npartials(FDNUM) == N
    @test ForwardDiff.npartials(typeof(FDNUM)) == N
    @test ForwardDiff.npartials(NESTED_FDNUM) == N
    @test ForwardDiff.npartials(typeof(NESTED_FDNUM)) == N

    @test ForwardDiff.valtype(FDNUM) == V
    @test ForwardDiff.valtype(typeof(FDNUM)) == V
    @test ForwardDiff.valtype(NESTED_FDNUM) == Dual{TestTag(),V,M}
    @test ForwardDiff.valtype(typeof(NESTED_FDNUM)) == Dual{TestTag(),V,M}

    #####################
    # Generic Functions #
    #####################

    @test FDNUM === copy(FDNUM)
    @test NESTED_FDNUM === copy(NESTED_FDNUM)

    if V != Int
        @test eps(FDNUM) === eps(PRIMAL)
        @test eps(typeof(FDNUM)) === eps(V)
        @test eps(NESTED_FDNUM) === eps(PRIMAL)
        @test eps(typeof(NESTED_FDNUM)) === eps(V)

        @test floor(Int, FDNUM) === floor(Int, PRIMAL)
        @test floor(Int, FDNUM2) === floor(Int, PRIMAL2)
        @test floor(Int, NESTED_FDNUM) === floor(Int, PRIMAL)

        @test floor(FDNUM) === floor(PRIMAL)
        @test floor(FDNUM2) === floor(PRIMAL2)
        @test floor(NESTED_FDNUM) === floor(PRIMAL)

        @test ceil(Int, FDNUM) === ceil(Int, PRIMAL)
        @test ceil(Int, FDNUM2) === ceil(Int, PRIMAL2)
        @test ceil(Int, NESTED_FDNUM) === ceil(Int, PRIMAL)

        @test ceil(FDNUM) === ceil(PRIMAL)
        @test ceil(FDNUM2) === ceil(PRIMAL2)
        @test ceil(NESTED_FDNUM) === ceil(PRIMAL)

        @test trunc(Int, FDNUM) === trunc(Int, PRIMAL)
        @test trunc(Int, FDNUM2) === trunc(Int, PRIMAL2)
        @test trunc(Int, NESTED_FDNUM) === trunc(Int, PRIMAL)

        @test trunc(FDNUM) === trunc(PRIMAL)
        @test trunc(FDNUM2) === trunc(PRIMAL2)
        @test trunc(NESTED_FDNUM) === trunc(PRIMAL)

        @test round(Int, FDNUM) === round(Int, PRIMAL)
        @test round(Int, FDNUM2) === round(Int, PRIMAL2)
        @test round(Int, NESTED_FDNUM) === round(Int, PRIMAL)

        @test round(FDNUM) === round(PRIMAL)
        @test round(FDNUM2) === round(PRIMAL2)
        @test round(NESTED_FDNUM) === round(PRIMAL)

        @test div(FDNUM, FDNUM2) === div(PRIMAL, PRIMAL2)
        @test div(FDNUM, PRIMAL2) === div(PRIMAL, PRIMAL2)
        @test div(PRIMAL, FDNUM2) === div(PRIMAL, PRIMAL2)

        @test div(NESTED_FDNUM, NESTED_FDNUM2) === div(PRIMAL, PRIMAL2)
        @test div(NESTED_FDNUM, PRIMAL2) === div(PRIMAL, PRIMAL2)
        @test div(PRIMAL, NESTED_FDNUM2) === div(PRIMAL, PRIMAL2)

        if VERSION ≥ v"1.4"
            @test div(FDNUM, FDNUM2, RoundUp) === div(PRIMAL, PRIMAL2, RoundUp)
            @test div(NESTED_FDNUM, NESTED_FDNUM2, RoundUp) === div(PRIMAL, PRIMAL2, RoundUp)
        end

        @test Base.rtoldefault(typeof(FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
        @test Dual{TestTag()}(PRIMAL-eps(V), PARTIALS) ≈ FDNUM
        @test Base.rtoldefault(typeof(NESTED_FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
    end

    @test hash(FDNUM) === hash(PRIMAL)
    @test hash(FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))
    @test hash(NESTED_FDNUM) === hash(PRIMAL)
    @test hash(NESTED_FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))

    TMPIO = IOBuffer()
    write(TMPIO, FDNUM)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(FDNUM)) === FDNUM
    seekstart(TMPIO)
    write(TMPIO, FDNUM2)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(FDNUM2)) === FDNUM2
    seekstart(TMPIO)
    write(TMPIO, NESTED_FDNUM)
    seekstart(TMPIO)
    @test read(TMPIO, typeof(NESTED_FDNUM)) === NESTED_FDNUM
    close(TMPIO)

    @test zero(FDNUM) === Dual{TestTag()}(zero(PRIMAL), zero(PARTIALS))
    @test zero(typeof(FDNUM)) === Dual{TestTag()}(zero(V), zero(Partials{N,V}))
    @test zero(NESTED_FDNUM) === Dual{TestTag()}(Dual{TestTag()}(zero(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test zero(typeof(NESTED_FDNUM)) === Dual{TestTag()}(Dual{TestTag()}(zero(V), zero(Partials{M,V})), zero(Partials{N,Dual{TestTag(),V,M}}))

    @test one(FDNUM) === Dual{TestTag()}(one(PRIMAL), zero(PARTIALS))
    @test one(typeof(FDNUM)) === Dual{TestTag()}(one(V), zero(Partials{N,V}))
    @test one(NESTED_FDNUM) === Dual{TestTag()}(Dual{TestTag()}(one(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test one(typeof(NESTED_FDNUM)) === Dual{TestTag()}(Dual{TestTag()}(one(V), zero(Partials{M,V})), zero(Partials{N,Dual{TestTag(),V,M}}))

    if V <: Integer
        @test rand(samerng(), FDNUM) == rand(samerng(), value(FDNUM))
        @test rand(samerng(), NESTED_FDNUM) == rand(samerng(), value(NESTED_FDNUM))
    elseif V <: AbstractFloat
        @test rand(samerng(), typeof(FDNUM)) === Dual{TestTag()}(rand(samerng(), V), zero(Partials{N,V}))
        @test rand(samerng(), typeof(NESTED_FDNUM)) === Dual{TestTag()}(Dual{TestTag()}(rand(samerng(), V), zero(Partials{M,V})), zero(Partials{N,Dual{TestTag(),V,M}}))
        @test randn(samerng(), typeof(FDNUM)) === Dual{TestTag()}(randn(samerng(), V), zero(Partials{N,V}))
        @test randn(samerng(), typeof(NESTED_FDNUM)) === Dual{TestTag()}(Dual{TestTag()}(randn(samerng(), V), zero(Partials{M,V})),
        zero(Partials{N,Dual{TestTag(),V,M}}))
        @test randexp(samerng(), typeof(FDNUM)) === Dual{TestTag()}(randexp(samerng(), V), zero(Partials{N,V}))
        @test randexp(samerng(), typeof(NESTED_FDNUM)) === Dual{TestTag()}(Dual{TestTag()}(randexp(samerng(), V), zero(Partials{M,V})),
        zero(Partials{N,Dual{TestTag(),V,M}}))
    end

    # Predicates #
    #------------#

    @test ForwardDiff.isconstant(zero(FDNUM))
    @test ForwardDiff.isconstant(one(FDNUM))
    @test ForwardDiff.isconstant(FDNUM) == (N == 0)

    @test ForwardDiff.isconstant(zero(NESTED_FDNUM))
    @test ForwardDiff.isconstant(one(NESTED_FDNUM))
    @test ForwardDiff.isconstant(NESTED_FDNUM) == (N == 0)

    @test isequal(FDNUM, Dual{TestTag()}(PRIMAL, PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(FDNUM, FDNUM2)

    @test isequal(NESTED_FDNUM, Dual{TestTag()}(Dual{TestTag()}(PRIMAL, M_PARTIALS2), NESTED_PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(NESTED_FDNUM, NESTED_FDNUM2)

    @test FDNUM == Dual{TestTag()}(PRIMAL, PARTIALS2)
    @test (PRIMAL == PRIMAL2) == (FDNUM == FDNUM2)
    @test (PRIMAL == PRIMAL2) == (NESTED_FDNUM == NESTED_FDNUM2)

    @test isless(Dual{TestTag()}(1, PARTIALS), Dual{TestTag()}(2, PARTIALS2))
    @test !(isless(Dual{TestTag()}(1, PARTIALS), Dual{TestTag()}(1, PARTIALS2)))
    @test !(isless(Dual{TestTag()}(2, PARTIALS), Dual{TestTag()}(1, PARTIALS2)))

    @test isless(Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS), Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(isless(Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS), Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2)))
    @test !(isless(Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS), NESTED_PARTIALS), Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2)))

    @test Dual{TestTag()}(1, PARTIALS) < Dual{TestTag()}(2, PARTIALS2)
    @test !(Dual{TestTag()}(1, PARTIALS) < Dual{TestTag()}(1, PARTIALS2))
    @test !(Dual{TestTag()}(2, PARTIALS) < Dual{TestTag()}(1, PARTIALS2))

    @test Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) < Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) < Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS), NESTED_PARTIALS) < Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual{TestTag()}(1, PARTIALS) <= Dual{TestTag()}(2, PARTIALS2)
    @test Dual{TestTag()}(1, PARTIALS) <= Dual{TestTag()}(1, PARTIALS2)
    @test !(Dual{TestTag()}(2, PARTIALS) <= Dual{TestTag()}(1, PARTIALS2))

    @test Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) <= Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS2), NESTED_PARTIALS2)
    @test Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) <= Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS), NESTED_PARTIALS) <= Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual{TestTag()}(2, PARTIALS) > Dual{TestTag()}(1, PARTIALS2)
    @test !(Dual{TestTag()}(1, PARTIALS) > Dual{TestTag()}(1, PARTIALS2))
    @test !(Dual{TestTag()}(1, PARTIALS) > Dual{TestTag()}(2, PARTIALS2))

    @test Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS), NESTED_PARTIALS) > Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) > Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) > Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual{TestTag()}(2, PARTIALS) >= Dual{TestTag()}(1, PARTIALS2)
    @test Dual{TestTag()}(1, PARTIALS) >= Dual{TestTag()}(1, PARTIALS2)
    @test !(Dual{TestTag()}(1, PARTIALS) >= Dual{TestTag()}(2, PARTIALS2))

    @test Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS), NESTED_PARTIALS) >= Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) >= Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual{TestTag()}(Dual{TestTag()}(1, M_PARTIALS), NESTED_PARTIALS) >= Dual{TestTag()}(Dual{TestTag()}(2, M_PARTIALS2), NESTED_PARTIALS2))

    @test isnan(Dual{TestTag()}(NaN, PARTIALS))
    @test !(isnan(FDNUM))

    @test isnan(Dual{TestTag()}(Dual{TestTag()}(NaN, M_PARTIALS), NESTED_PARTIALS))
    @test !(isnan(NESTED_FDNUM))

    @test isfinite(FDNUM)
    @test !(isfinite(Dual{TestTag()}(Inf, PARTIALS)))

    @test isfinite(NESTED_FDNUM)
    @test !(isfinite(Dual{TestTag()}(Dual{TestTag()}(NaN, M_PARTIALS), NESTED_PARTIALS)))

    @test isinf(Dual{TestTag()}(Inf, PARTIALS))
    @test !(isinf(FDNUM))

    @test isinf(Dual{TestTag()}(Dual{TestTag()}(Inf, M_PARTIALS), NESTED_PARTIALS))
    @test !(isinf(NESTED_FDNUM))

    @test isreal(FDNUM)
    @test isreal(NESTED_FDNUM)

    @test isinteger(Dual{TestTag()}(1.0, PARTIALS))
    @test isinteger(FDNUM) == (V == Int)

    @test isinteger(Dual{TestTag()}(Dual{TestTag()}(1.0, M_PARTIALS), NESTED_PARTIALS))
    @test isinteger(NESTED_FDNUM) == (V == Int)

    @test iseven(Dual{TestTag()}(2))
    @test !(iseven(Dual{TestTag()}(1)))

    @test iseven(Dual{TestTag()}(Dual{TestTag()}(2)))
    @test !(iseven(Dual{TestTag()}(Dual{TestTag()}(1))))

    @test isodd(Dual{TestTag()}(1))
    @test !(isodd(Dual{TestTag()}(2)))

    @test isodd(Dual{TestTag()}(Dual{TestTag()}(1)))
    @test !(isodd(Dual{TestTag()}(Dual{TestTag()}(2))))

    ########################
    # Promotion/Conversion #
    ########################

    WIDE_T = widen(V)

    @test promote_type(Dual{TestTag(),V,N}, V) == Dual{TestTag(),V,N}
    @test promote_type(Dual{TestTag(),V,N}, WIDE_T) == Dual{TestTag(),WIDE_T,N}
    @test promote_type(Dual{TestTag(),WIDE_T,N}, V) == Dual{TestTag(),WIDE_T,N}
    @test promote_type(Dual{TestTag(),V,N}, Dual{TestTag(),V,N}) == Dual{TestTag(),V,N}
    @test promote_type(Dual{TestTag(),V,N}, Dual{TestTag(),WIDE_T,N}) == Dual{TestTag(),WIDE_T,N}
    @test promote_type(Dual{TestTag(),WIDE_T,N}, Dual{TestTag(),Dual{TestTag(),V,M},N}) == Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}

    # issue #322
    @test promote_type(Bool, Dual{TestTag(),V,N}) == Dual{TestTag(),promote_type(Bool, V),N}
    @test promote_type(BigFloat, Dual{TestTag(),V,N}) == Dual{TestTag(),promote_type(BigFloat, V),N}

    WIDE_FDNUM = convert(Dual{TestTag(),WIDE_T,N}, FDNUM)
    WIDE_NESTED_FDNUM = convert(Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}, NESTED_FDNUM)

    @test typeof(WIDE_FDNUM) === Dual{TestTag(),WIDE_T,N}
    @test typeof(WIDE_NESTED_FDNUM) === Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}

    @test value(WIDE_FDNUM) == PRIMAL
    @test value(WIDE_NESTED_FDNUM) == PRIMAL

    @test convert(Dual, FDNUM) === FDNUM
    @test convert(Dual, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{TestTag(),V,N}, FDNUM) === FDNUM
    @test convert(Dual{TestTag(),Dual{TestTag(),V,M},N}, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{TestTag(),WIDE_T,N}, PRIMAL) === Dual{TestTag()}(WIDE_T(PRIMAL), zero(Partials{N,WIDE_T}))
    @test convert(Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}, PRIMAL) === Dual{TestTag()}(Dual{TestTag()}(WIDE_T(PRIMAL), zero(Partials{M,WIDE_T})), zero(Partials{N,Dual{TestTag(),V,M}}))
    @test convert(Dual{TestTag(),Dual{TestTag(),V,M},N}, FDNUM) === Dual{TestTag()}(convert(Dual{TestTag(),V,M}, PRIMAL), convert(Partials{N,Dual{TestTag(),V,M}}, PARTIALS))
    @test convert(Dual{TestTag(),Dual{TestTag(),WIDE_T,M},N}, FDNUM) === Dual{TestTag()}(convert(Dual{TestTag(),WIDE_T,M}, PRIMAL), convert(Partials{N,Dual{TestTag(),WIDE_T,M}}, PARTIALS))

    ##############
    # Arithmetic #
    ##############

    # Addition/Subtraction #
    #----------------------#

    @test FDNUM + FDNUM2 === Dual{TestTag()}(value(FDNUM) + value(FDNUM2), partials(FDNUM) + partials(FDNUM2))
    @test FDNUM + PRIMAL === Dual{TestTag()}(value(FDNUM) + PRIMAL, partials(FDNUM))
    @test PRIMAL + FDNUM === Dual{TestTag()}(value(FDNUM) + PRIMAL, partials(FDNUM))

    @test NESTED_FDNUM + NESTED_FDNUM2 === Dual{TestTag()}(value(NESTED_FDNUM) + value(NESTED_FDNUM2), partials(NESTED_FDNUM) + partials(NESTED_FDNUM2))
    @test NESTED_FDNUM + PRIMAL === Dual{TestTag()}(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))
    @test PRIMAL + NESTED_FDNUM === Dual{TestTag()}(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))

    @test FDNUM - FDNUM2 === Dual{TestTag()}(value(FDNUM) - value(FDNUM2), partials(FDNUM) - partials(FDNUM2))
    @test FDNUM - PRIMAL === Dual{TestTag()}(value(FDNUM) - PRIMAL, partials(FDNUM))
    @test PRIMAL - FDNUM === Dual{TestTag()}(PRIMAL - value(FDNUM), -(partials(FDNUM)))
    @test -(FDNUM) === Dual{TestTag()}(-(value(FDNUM)), -(partials(FDNUM)))

    @test NESTED_FDNUM - NESTED_FDNUM2 === Dual{TestTag()}(value(NESTED_FDNUM) - value(NESTED_FDNUM2), partials(NESTED_FDNUM) - partials(NESTED_FDNUM2))
    @test NESTED_FDNUM - PRIMAL === Dual{TestTag()}(value(NESTED_FDNUM) - PRIMAL, partials(NESTED_FDNUM))
    @test PRIMAL - NESTED_FDNUM === Dual{TestTag()}(PRIMAL - value(NESTED_FDNUM), -(partials(NESTED_FDNUM)))
    @test -(NESTED_FDNUM) === Dual{TestTag()}(-(value(NESTED_FDNUM)), -(partials(NESTED_FDNUM)))

    # Multiplication #
    #----------------#

    @test FDNUM * FDNUM2 === Dual{TestTag()}(value(FDNUM) * value(FDNUM2), ForwardDiff._mul_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM2), value(FDNUM)))
    @test FDNUM * PRIMAL === Dual{TestTag()}(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)
    @test PRIMAL * FDNUM === Dual{TestTag()}(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)

    @test NESTED_FDNUM * NESTED_FDNUM2 === Dual{TestTag()}(value(NESTED_FDNUM) * value(NESTED_FDNUM2), ForwardDiff._mul_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM2), value(NESTED_FDNUM)))
    @test NESTED_FDNUM * PRIMAL === Dual{TestTag()}(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)
    @test PRIMAL * NESTED_FDNUM === Dual{TestTag()}(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)

    # Division #
    #----------#

    if M > 0 && N > 0
        @test Dual{1}(FDNUM) / Dual{1}(PRIMAL) === Dual{1}(FDNUM / PRIMAL)
        @test Dual{1}(PRIMAL) / Dual{1}(FDNUM) === Dual{1}(PRIMAL / FDNUM)
        @test_broken Dual{1}(FDNUM) / FDNUM2 === Dual{1}(FDNUM / FDNUM2)
        @test_broken FDNUM / Dual{1}(FDNUM2) === Dual{1}(FDNUM / FDNUM2)
        # following may not be exact, see #264
        @test Dual{1}(FDNUM / PRIMAL, FDNUM2 / PRIMAL) ≈ Dual{1}(FDNUM, FDNUM2) / PRIMAL
    end

    @test dual_isapprox(FDNUM / FDNUM2, Dual{TestTag()}(value(FDNUM) / value(FDNUM2), ForwardDiff._div_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM), value(FDNUM2))))
    @test dual_isapprox(FDNUM / PRIMAL, Dual{TestTag()}(value(FDNUM) / PRIMAL, partials(FDNUM) / PRIMAL))
    @test dual_isapprox(PRIMAL / FDNUM, Dual{TestTag()}(PRIMAL / value(FDNUM), (-(PRIMAL) / value(FDNUM)^2) * partials(FDNUM)))

    @test dual_isapprox(NESTED_FDNUM / NESTED_FDNUM2, Dual{TestTag()}(value(NESTED_FDNUM) / value(NESTED_FDNUM2), ForwardDiff._div_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM), value(NESTED_FDNUM2))))
    @test dual_isapprox(NESTED_FDNUM / PRIMAL, Dual{TestTag()}(value(NESTED_FDNUM) / PRIMAL, partials(NESTED_FDNUM) / PRIMAL))
    @test dual_isapprox(PRIMAL / NESTED_FDNUM, Dual{TestTag()}(PRIMAL / value(NESTED_FDNUM), (-(PRIMAL) / value(NESTED_FDNUM)^2) * partials(NESTED_FDNUM)))

    # Exponentiation #
    #----------------#

    @test dual_isapprox(FDNUM^FDNUM2, exp(FDNUM2 * log(FDNUM)))
    @test dual_isapprox(FDNUM^PRIMAL, exp(PRIMAL * log(FDNUM)))
    @test dual_isapprox(PRIMAL^FDNUM, exp(FDNUM * log(PRIMAL)))

    @test dual_isapprox(NESTED_FDNUM^NESTED_FDNUM2, exp(NESTED_FDNUM2 * log(NESTED_FDNUM)))
    @test dual_isapprox(NESTED_FDNUM^PRIMAL, exp(PRIMAL * log(NESTED_FDNUM)))
    @test dual_isapprox(PRIMAL^NESTED_FDNUM, exp(NESTED_FDNUM * log(PRIMAL)))

    @test partials(NaNMath.pow(Dual{TestTag()}(-2.0, 1.0), Dual{TestTag()}(2.0, 0.0)), 1) == -4.0

    ###################################
    # General Mathematical Operations #
    ###################################

    @test conj(FDNUM) === FDNUM
    @test conj(NESTED_FDNUM) === NESTED_FDNUM

    @test transpose(FDNUM) === FDNUM
    @test transpose(NESTED_FDNUM) === NESTED_FDNUM

    @test abs(-FDNUM) === FDNUM
    @test abs(FDNUM) === FDNUM
    @test abs(-NESTED_FDNUM) === NESTED_FDNUM
    @test abs(NESTED_FDNUM) === NESTED_FDNUM

    if V != Int
        for (M, f, arity) in DiffRules.diffrules()
            in(f, (:hankelh1, :hankelh1x, :hankelh2, :hankelh2x, :/, :rem2pi)) && continue
            println("       ...auto-testing $(M).$(f) with $arity arguments")
            if arity == 1
                deriv = DiffRules.diffrule(M, f, :x)
                modifier = in(f, (:asec, :acsc, :asecd, :acscd, :acosh, :acoth)) ? one(V) : zero(V)
                @eval begin
                    x = rand() + $modifier
                    dx = $M.$f(Dual{TestTag()}(x, one(x)))
                    @test value(dx) == $M.$f(x)
                    @test partials(dx, 1) == $deriv
                end
            elseif arity == 2
                derivs = DiffRules.diffrule(M, f, :x, :y)
                @eval begin
                    x, y = rand(1:10), rand()
                    dx = $M.$f(Dual{TestTag()}(x, one(x)), y)
                    dy = $M.$f(x, Dual{TestTag()}(y, one(y)))
                    actualdx = $(derivs[1])
                    actualdy = $(derivs[2])
                    @test value(dx) == $M.$f(x, y)
                    @test value(dy) == value(dx)
                    if isnan(actualdx)
                        @test isnan(partials(dx, 1))
                    else
                        @test partials(dx, 1) ≈ actualdx
                    end
                    if isnan(actualdy)
                        @test isnan(partials(dy, 1))
                    else
                        @test partials(dy, 1) ≈ actualdy
                    end
                end
            end
        end
    end

    # Special Cases #
    #---------------#

    @test_broken dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM), sqrt(2*(FDNUM^2) + FDNUM2^2))
    @test_broken dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM3), sqrt(FDNUM^2 + FDNUM2^2 + FDNUM3^2))

    @test all(map(dual_isapprox, ForwardDiff.sincos(FDNUM), (sin(FDNUM), cos(FDNUM))))

    if VERSION >= v"1.6.0-DEV.292"
        @test all(map(dual_isapprox, sincospi(FDNUM), (sinpi(FDNUM), cospi(FDNUM))))
    end

    if V === Float32
        @test typeof(sqrt(FDNUM)) === typeof(FDNUM)
        @test typeof(sqrt(NESTED_FDNUM)) === typeof(NESTED_FDNUM)
    end

    for f in (fma, muladd)
        @test dual_isapprox(f(FDNUM, FDNUM2, FDNUM3),   Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS + PARTIALS3))
        @test dual_isapprox(f(FDNUM, FDNUM2, PRIMAL3),  Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS))
        @test dual_isapprox(f(PRIMAL, FDNUM2, FDNUM3),  Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2 + PARTIALS3))
        @test dual_isapprox(f(PRIMAL, FDNUM2, PRIMAL3), Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL*PARTIALS2))
        @test dual_isapprox(f(FDNUM, PRIMAL2, FDNUM3),  Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL2*PARTIALS + PARTIALS3))
        @test dual_isapprox(f(FDNUM, PRIMAL2, PRIMAL3), Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PRIMAL2*PARTIALS))
        @test dual_isapprox(f(PRIMAL, PRIMAL2, FDNUM3), Dual{TestTag()}(f(PRIMAL, PRIMAL2, PRIMAL3), PARTIALS3))
    end
end

@testset "Exponentiation of zero" begin
    x0 = 0.0
    x1 = Dual{:t1}(x0, 1.0)
    x2 = Dual{:t2}(x1, 1.0)
    x3 = Dual{:t3}(x2, 1.0)
    pow = ^  # to call non-literal power
    @test pow(x3, 2) === x3^2 === x3 * x3
    @test pow(x2, 1) === x2^1 === x2
    @test pow(x1, 0) === x1^0 === Dual{:t1}(1.0, 0.0)
end

@testset "Type min/max" begin
    d1 = Dual(1.0)
    dinf = typemax(typeof(d1))
    dminf = typemin(typeof(d1))
    @test dminf < d1 < dinf
    @test typeof(dminf) === typeof(d1)
    @test typeof(dinf) === typeof(d1)
    @test !isfinite(dminf)
    @test !isfinite(dinf)
end

@testset "Integer" begin
    x = Dual(1.0,0)
    @test Int(x) ≡ Integer(x) ≡ convert(Int,x) ≡ convert(Integer,x) ≡ 1
    x = Dual(1.0,0,0)
    @test Int(x) ≡ Integer(x) ≡ convert(Int,x) ≡ convert(Integer,x) ≡ 1
    @test_throws InexactError Int(Dual(1.5,1.2))
    @test_throws InexactError Integer(Dual(1.5,1.2))
    @test_throws InexactError Int(Dual(1,1))
    @test_throws InexactError Integer(Dual(1,1,2))
    @test length(UnitRange(Dual(1.5), Dual(3.5))) == 3
    @test length(UnitRange(Dual(1.5,1), Dual(3.5,3))) == 3
end

end # module
