module DualTest

using Base.Test
using ForwardDiff
using ForwardDiff: Partials, Dual, value, partials

import NaNMath
import Calculus
import SpecialFunctions

samerng() = MersenneTwister(1)

# By lower-bounding the Int range at 2, we avoid cases where differentiating an
# exponentiation of an Int value would cause a DomainError due to reducing the
# exponent by one
intrand(V) = V == Int ? rand(2:10) : rand(V)

dual_isapprox(a, b) = isapprox(a, b)
dual_isapprox(a::Dual, b::Dual) = isapprox(value(a), value(b)) && isapprox(partials(a), partials(b))

for N in (0,3), M in (0,4), V in (Int, Float32)
    println("  ...testing Dual{Void,$V,$N} and Dual{Void,Dual{Void,$V,$M},$N}")

    PARTIALS = Partials{N,V}(ntuple(n -> intrand(V), Val{N}))
    PRIMAL = intrand(V)
    FDNUM = Dual(PRIMAL, PARTIALS)

    PARTIALS2 = Partials{N,V}(ntuple(n -> intrand(V), Val{N}))
    PRIMAL2 = intrand(V)
    FDNUM2 = Dual(PRIMAL2, PARTIALS2)

    PARTIALS3 = Partials{N,V}(ntuple(n -> intrand(V), Val{N}))
    PRIMAL3 = intrand(V)
    FDNUM3 = Dual(PRIMAL3, PARTIALS3)

    M_PARTIALS = Partials{M,V}(ntuple(m -> intrand(V), Val{M}))
    NESTED_PARTIALS = convert(Partials{N,Dual{Void,V,M}}, PARTIALS)
    NESTED_FDNUM = Dual(Dual(PRIMAL, M_PARTIALS), NESTED_PARTIALS)

    M_PARTIALS2 = Partials{M,V}(ntuple(m -> intrand(V), Val{M}))
    NESTED_PARTIALS2 = convert(Partials{N,Dual{Void,V,M}}, PARTIALS2)
    NESTED_FDNUM2 = Dual(Dual(PRIMAL2, M_PARTIALS2), NESTED_PARTIALS2)

    ################
    # Constructors #
    ################

    @test Dual(PRIMAL, PARTIALS...) === FDNUM
    @test typeof(Dual(widen(V)(PRIMAL), PARTIALS)) === Dual{Void,widen(V),N}
    @test typeof(Dual(widen(V)(PRIMAL), PARTIALS.values)) === Dual{Void,widen(V),N}
    @test typeof(Dual(widen(V)(PRIMAL), PARTIALS...)) === Dual{Void,widen(V),N}
    @test typeof(NESTED_FDNUM) == Dual{Void,Dual{Void,V,M},N}

    #############
    # Accessors #
    #############

    @test value(PRIMAL) == PRIMAL
    @test value(FDNUM) == PRIMAL
    @test value(NESTED_FDNUM) === Dual(PRIMAL, M_PARTIALS)

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
    @test ForwardDiff.valtype(NESTED_FDNUM) == Dual{Void,V,M}
    @test ForwardDiff.valtype(typeof(NESTED_FDNUM)) == Dual{Void,V,M}

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

        @test Base.rtoldefault(typeof(FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
        @test Dual(PRIMAL-eps(V), PARTIALS) ≈ FDNUM
        @test Base.rtoldefault(typeof(NESTED_FDNUM)) ≡ Base.rtoldefault(typeof(PRIMAL))
    end

    @test hash(FDNUM) === hash(PRIMAL)
    @test hash(FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))
    @test hash(NESTED_FDNUM) === hash(PRIMAL)
    @test hash(NESTED_FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))

    const TMPIO = IOBuffer()
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

    @test zero(FDNUM) === Dual(zero(PRIMAL), zero(PARTIALS))
    @test zero(typeof(FDNUM)) === Dual(zero(V), zero(Partials{N,V}))
    @test zero(NESTED_FDNUM) === Dual(Dual(zero(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test zero(typeof(NESTED_FDNUM)) === Dual(Dual(zero(V), zero(Partials{M,V})), zero(Partials{N,Dual{Void,V,M}}))

    @test one(FDNUM) === Dual(one(PRIMAL), zero(PARTIALS))
    @test one(typeof(FDNUM)) === Dual(one(V), zero(Partials{N,V}))
    @test one(NESTED_FDNUM) === Dual(Dual(one(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test one(typeof(NESTED_FDNUM)) === Dual(Dual(one(V), zero(Partials{M,V})), zero(Partials{N,Dual{Void,V,M}}))

    @test rand(samerng(), FDNUM) === Dual(rand(samerng(), V), zero(PARTIALS))
    @test rand(samerng(), typeof(FDNUM)) === Dual(rand(samerng(), V), zero(Partials{N,V}))
    @test rand(samerng(), NESTED_FDNUM) === Dual(Dual(rand(samerng(), V), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
    @test rand(samerng(), typeof(NESTED_FDNUM)) === Dual(Dual(rand(samerng(), V), zero(Partials{M,V})), zero(Partials{N,Dual{Void,V,M}}))

    # Predicates #
    #------------#

    @test ForwardDiff.isconstant(zero(FDNUM))
    @test ForwardDiff.isconstant(rand(FDNUM))
    @test ForwardDiff.isconstant(one(FDNUM))
    @test ForwardDiff.isconstant(FDNUM) == (N == 0)

    @test ForwardDiff.isconstant(zero(NESTED_FDNUM))
    @test ForwardDiff.isconstant(rand(NESTED_FDNUM))
    @test ForwardDiff.isconstant(one(NESTED_FDNUM))
    @test ForwardDiff.isconstant(NESTED_FDNUM) == (N == 0)

    @test isequal(FDNUM, Dual(PRIMAL, PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(FDNUM, FDNUM2)

    @test isequal(NESTED_FDNUM, Dual(Dual(PRIMAL, M_PARTIALS2), NESTED_PARTIALS2))
    @test isequal(PRIMAL, PRIMAL2) == isequal(NESTED_FDNUM, NESTED_FDNUM2)

    @test FDNUM == Dual(PRIMAL, PARTIALS2)
    @test (PRIMAL == PRIMAL2) == (FDNUM == FDNUM2)
    @test (PRIMAL == PRIMAL2) == (NESTED_FDNUM == NESTED_FDNUM2)

    @test isless(Dual(1, PARTIALS), Dual(2, PARTIALS2))
    @test !(isless(Dual(1, PARTIALS), Dual(1, PARTIALS2)))
    @test !(isless(Dual(2, PARTIALS), Dual(1, PARTIALS2)))

    @test isless(Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS), Dual(Dual(2, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(isless(Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS), Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2)))
    @test !(isless(Dual(Dual(2, M_PARTIALS), NESTED_PARTIALS), Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2)))

    @test Dual(1, PARTIALS) < Dual(2, PARTIALS2)
    @test !(Dual(1, PARTIALS) < Dual(1, PARTIALS2))
    @test !(Dual(2, PARTIALS) < Dual(1, PARTIALS2))

    @test Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) < Dual(Dual(2, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) < Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(Dual(Dual(2, M_PARTIALS), NESTED_PARTIALS) < Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual(1, PARTIALS) <= Dual(2, PARTIALS2)
    @test Dual(1, PARTIALS) <= Dual(1, PARTIALS2)
    @test !(Dual(2, PARTIALS) <= Dual(1, PARTIALS2))

    @test Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) <= Dual(Dual(2, M_PARTIALS2), NESTED_PARTIALS2)
    @test Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) <= Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual(Dual(2, M_PARTIALS), NESTED_PARTIALS) <= Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual(2, PARTIALS) > Dual(1, PARTIALS2)
    @test !(Dual(1, PARTIALS) > Dual(1, PARTIALS2))
    @test !(Dual(1, PARTIALS) > Dual(2, PARTIALS2))

    @test Dual(Dual(2, M_PARTIALS), NESTED_PARTIALS) > Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) > Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2))
    @test !(Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) > Dual(Dual(2, M_PARTIALS2), NESTED_PARTIALS2))

    @test Dual(2, PARTIALS) >= Dual(1, PARTIALS2)
    @test Dual(1, PARTIALS) >= Dual(1, PARTIALS2)
    @test !(Dual(1, PARTIALS) >= Dual(2, PARTIALS2))

    @test Dual(Dual(2, M_PARTIALS), NESTED_PARTIALS) >= Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) >= Dual(Dual(1, M_PARTIALS2), NESTED_PARTIALS2)
    @test !(Dual(Dual(1, M_PARTIALS), NESTED_PARTIALS) >= Dual(Dual(2, M_PARTIALS2), NESTED_PARTIALS2))

    @test isnan(Dual(NaN, PARTIALS))
    @test !(isnan(FDNUM))

    @test isnan(Dual(Dual(NaN, M_PARTIALS), NESTED_PARTIALS))
    @test !(isnan(NESTED_FDNUM))

    @test isfinite(FDNUM)
    @test !(isfinite(Dual(Inf, PARTIALS)))

    @test isfinite(NESTED_FDNUM)
    @test !(isfinite(Dual(Dual(NaN, M_PARTIALS), NESTED_PARTIALS)))

    @test isinf(Dual(Inf, PARTIALS))
    @test !(isinf(FDNUM))

    @test isinf(Dual(Dual(Inf, M_PARTIALS), NESTED_PARTIALS))
    @test !(isinf(NESTED_FDNUM))

    @test isreal(FDNUM)
    @test isreal(NESTED_FDNUM)

    @test isinteger(Dual(1.0, PARTIALS))
    @test isinteger(FDNUM) == (V == Int)

    @test isinteger(Dual(Dual(1.0, M_PARTIALS), NESTED_PARTIALS))
    @test isinteger(NESTED_FDNUM) == (V == Int)

    @test iseven(Dual(2))
    @test !(iseven(Dual(1)))

    @test iseven(Dual(Dual(2)))
    @test !(iseven(Dual(Dual(1))))

    @test isodd(Dual(1))
    @test !(isodd(Dual(2)))

    @test isodd(Dual(Dual(1)))
    @test !(isodd(Dual(Dual(2))))

    ########################
    # Promotion/Conversion #
    ########################

    WIDE_T = widen(V)

    @test promote_type(Dual{Void,V,N}, V) == Dual{Void,V,N}
    @test promote_type(Dual{Void,V,N}, WIDE_T) == Dual{Void,WIDE_T,N}
    @test promote_type(Dual{Void,WIDE_T,N}, V) == Dual{Void,WIDE_T,N}
    @test promote_type(Dual{Void,V,N}, Dual{Void,V,N}) == Dual{Void,V,N}
    @test promote_type(Dual{Void,V,N}, Dual{Void,WIDE_T,N}) == Dual{Void,WIDE_T,N}
    @test promote_type(Dual{Void,WIDE_T,N}, Dual{Void,Dual{Void,V,M},N}) == Dual{Void,Dual{Void,WIDE_T,M},N}

    WIDE_FDNUM = convert(Dual{Void,WIDE_T,N}, FDNUM)
    WIDE_NESTED_FDNUM = convert(Dual{Void,Dual{Void,WIDE_T,M},N}, NESTED_FDNUM)

    @test typeof(WIDE_FDNUM) === Dual{Void,WIDE_T,N}
    @test typeof(WIDE_NESTED_FDNUM) === Dual{Void,Dual{Void,WIDE_T,M},N}

    @test value(WIDE_FDNUM) == PRIMAL
    @test value(WIDE_NESTED_FDNUM) == PRIMAL

    @test convert(Dual, FDNUM) === FDNUM
    @test convert(Dual, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{Void,V,N}, FDNUM) === FDNUM
    @test convert(Dual{Void,Dual{Void,V,M},N}, NESTED_FDNUM) === NESTED_FDNUM
    @test convert(Dual{Void,WIDE_T,N}, PRIMAL) === Dual(WIDE_T(PRIMAL), zero(Partials{N,WIDE_T}))
    @test convert(Dual{Void,Dual{Void,WIDE_T,M},N}, PRIMAL) === Dual(Dual(WIDE_T(PRIMAL), zero(Partials{M,WIDE_T})), zero(Partials{N,Dual{Void,V,M}}))
    @test convert(Dual{Void,Dual{Void,V,M},N}, FDNUM) === Dual(Dual{Void,V,M}(PRIMAL), convert(Partials{N,Dual{Void,V,M}}, PARTIALS))
    @test convert(Dual{Void,Dual{Void,WIDE_T,M},N}, FDNUM) === Dual(Dual{Void,WIDE_T,M}(PRIMAL), convert(Partials{N,Dual{Void,WIDE_T,M}}, PARTIALS))

    if V != Int
        @test Base.promote_array_type(+, Dual{Void,V,N}, V, Base.promote_op(+, Dual{Void,V,N}, V)) == Dual{Void,V,N}
        @test Base.promote_array_type(+, Dual{Void,Int,N}, V, Base.promote_op(+, Dual{Void,Int,N}, V)) == Dual{Void,V,N}
        @test Base.promote_array_type(+, V, Dual{Void,V,N}, Base.promote_op(+, V, Dual{Void,V,N})) == Dual{Void,V,N}
        @test Base.promote_array_type(+, V, Dual{Void,Int,N}, Base.promote_op(+, V, Dual{Void,Int,N})) == Dual{Void,V,N}
        @test Base.promote_array_type(+, Dual{Void,V,N}, V) == Dual{Void,V,N}
        @test Base.promote_array_type(+, Dual{Void,Int,N}, V) == Dual{Void,V,N}
        @test Base.promote_array_type(+, V, Dual{Void,V,N}) == Dual{Void,V,N}
        @test Base.promote_array_type(+, V, Dual{Void,Int,N}) == Dual{Void,V,N}
    end

    ########
    # Math #
    ########

    # Arithmetic #
    #------------#

    @test FDNUM + FDNUM2 === Dual(value(FDNUM) + value(FDNUM2), partials(FDNUM) + partials(FDNUM2))
    @test FDNUM + PRIMAL === Dual(value(FDNUM) + PRIMAL, partials(FDNUM))
    @test PRIMAL + FDNUM === Dual(value(FDNUM) + PRIMAL, partials(FDNUM))

    @test NESTED_FDNUM + NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) + value(NESTED_FDNUM2), partials(NESTED_FDNUM) + partials(NESTED_FDNUM2))
    @test NESTED_FDNUM + PRIMAL === Dual(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))
    @test PRIMAL + NESTED_FDNUM === Dual(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))

    @test FDNUM - FDNUM2 === Dual(value(FDNUM) - value(FDNUM2), partials(FDNUM) - partials(FDNUM2))
    @test FDNUM - PRIMAL === Dual(value(FDNUM) - PRIMAL, partials(FDNUM))
    @test PRIMAL - FDNUM === Dual(PRIMAL - value(FDNUM), -(partials(FDNUM)))
    @test -(FDNUM) === Dual(-(value(FDNUM)), -(partials(FDNUM)))

    @test NESTED_FDNUM - NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) - value(NESTED_FDNUM2), partials(NESTED_FDNUM) - partials(NESTED_FDNUM2))
    @test NESTED_FDNUM - PRIMAL === Dual(value(NESTED_FDNUM) - PRIMAL, partials(NESTED_FDNUM))
    @test PRIMAL - NESTED_FDNUM === Dual(PRIMAL - value(NESTED_FDNUM), -(partials(NESTED_FDNUM)))
    @test -(NESTED_FDNUM) === Dual(-(value(NESTED_FDNUM)), -(partials(NESTED_FDNUM)))

    @test FDNUM * FDNUM2 === Dual(value(FDNUM) * value(FDNUM2), ForwardDiff._mul_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM2), value(FDNUM)))
    @test FDNUM * PRIMAL === Dual(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)
    @test PRIMAL * FDNUM === Dual(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)

    @test NESTED_FDNUM * NESTED_FDNUM2 === Dual(value(NESTED_FDNUM) * value(NESTED_FDNUM2), ForwardDiff._mul_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM2), value(NESTED_FDNUM)))
    @test NESTED_FDNUM * PRIMAL === Dual(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)
    @test PRIMAL * NESTED_FDNUM === Dual(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)

    if M > 0 && N > 0
        @test Dual{1}(FDNUM) / Dual{1}(PRIMAL) === Dual{1}(FDNUM / PRIMAL)
        @test Dual{1}(PRIMAL) / Dual{1}(FDNUM) === Dual{1}(PRIMAL / FDNUM)
        @test Dual{1}(FDNUM) / FDNUM2 === Dual{1}(FDNUM / FDNUM2)
        @test FDNUM / Dual{1}(FDNUM2) === Dual{1}(FDNUM / FDNUM2)
        @test Dual{1}(FDNUM / PRIMAL, FDNUM2 / PRIMAL) === Dual{1}(FDNUM, FDNUM2) / PRIMAL
    end

    @test dual_isapprox(FDNUM / FDNUM2, Dual(value(FDNUM) / value(FDNUM2), ForwardDiff._div_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM), value(FDNUM2))))
    @test dual_isapprox(FDNUM / PRIMAL, Dual(value(FDNUM) / PRIMAL, partials(FDNUM) / PRIMAL))
    @test dual_isapprox(PRIMAL / FDNUM, Dual(PRIMAL / value(FDNUM), (-(PRIMAL) / value(FDNUM)^2) * partials(FDNUM)))

    @test dual_isapprox(NESTED_FDNUM / NESTED_FDNUM2, Dual(value(NESTED_FDNUM) / value(NESTED_FDNUM2), ForwardDiff._div_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM), value(NESTED_FDNUM2))))
    @test dual_isapprox(NESTED_FDNUM / PRIMAL, Dual(value(NESTED_FDNUM) / PRIMAL, partials(NESTED_FDNUM) / PRIMAL))
    @test dual_isapprox(PRIMAL / NESTED_FDNUM, Dual(PRIMAL / value(NESTED_FDNUM), (-(PRIMAL) / value(NESTED_FDNUM)^2) * partials(NESTED_FDNUM)))

    @test dual_isapprox(FDNUM^FDNUM2, exp(FDNUM2 * log(FDNUM)))
    @test dual_isapprox(FDNUM^PRIMAL, exp(PRIMAL * log(FDNUM)))
    @test dual_isapprox(PRIMAL^FDNUM, exp(FDNUM * log(PRIMAL)))

    @test dual_isapprox(NESTED_FDNUM^NESTED_FDNUM2, exp(NESTED_FDNUM2 * log(NESTED_FDNUM)))
    @test dual_isapprox(NESTED_FDNUM^PRIMAL, exp(PRIMAL * log(NESTED_FDNUM)))
    @test dual_isapprox(PRIMAL^NESTED_FDNUM, exp(NESTED_FDNUM * log(PRIMAL)))

    @test partials(NaNMath.pow(Dual(-2.0, 1.0), Dual(2.0, 0.0)), 1) == -4.0

    test_approx_diffnums(fma(FDNUM, FDNUM2, FDNUM3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3),
                                             PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS +
                                             PARTIALS3))
    test_approx_diffnums(fma(FDNUM, FDNUM2, PRIMAL3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3),
                                              PRIMAL*PARTIALS2 + PRIMAL2*PARTIALS))
    test_approx_diffnums(fma(PRIMAL, FDNUM2, FDNUM3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3),
                                              PRIMAL*PARTIALS2 + PARTIALS3))
    test_approx_diffnums(fma(PRIMAL, FDNUM2, PRIMAL3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3),
                                               PRIMAL*PARTIALS2))
    test_approx_diffnums(fma(FDNUM, PRIMAL2, FDNUM3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3),
                                              PRIMAL2*PARTIALS + PARTIALS3))
    test_approx_diffnums(fma(FDNUM, PRIMAL2, PRIMAL3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3),
                                               PRIMAL2*PARTIALS))
    test_approx_diffnums(fma(PRIMAL, PRIMAL2, FDNUM3), Dual(fma(PRIMAL, PRIMAL2, PRIMAL3), PARTIALS3))

    # Unary Functions #
    #-----------------#

    @test conj(FDNUM) === FDNUM
    @test conj(NESTED_FDNUM) === NESTED_FDNUM
    @test transpose(FDNUM) === FDNUM
    @test transpose(NESTED_FDNUM) === NESTED_FDNUM
    @test ctranspose(FDNUM) === FDNUM
    @test ctranspose(NESTED_FDNUM) === NESTED_FDNUM

    @test abs(-FDNUM) === FDNUM
    @test abs(FDNUM) === FDNUM
    @test abs(-NESTED_FDNUM) === NESTED_FDNUM
    @test abs(NESTED_FDNUM) === NESTED_FDNUM

    if V != Int
        UNSUPPORTED_NESTED_FUNCS = (:trigamma, :airyprime, :besselj1, :bessely1)
        DOMAIN_ERR_FUNCS = (:asec, :acsc, :asecd, :acscd, :acoth, :acosh)

        for fsym in ForwardDiff.AUTO_DEFINED_UNARY_FUNCS
            try
                v = :v
                deriv = Calculus.differentiate(:($(fsym)($v)), v)
                is_nanmath_func = in(fsym, ForwardDiff.NANMATH_FUNCS)
                is_special_func = in(fsym, ForwardDiff.SPECIAL_FUNCS)
                is_domain_err_func = in(fsym, DOMAIN_ERR_FUNCS)
                is_unsupported_nested_func = in(fsym, UNSUPPORTED_NESTED_FUNCS)
                tested_funcs = Vector{Expr}(0)
                is_nanmath_func && push!(tested_funcs, :(NaNMath.$(fsym)))
                is_special_func && push!(tested_funcs, :(SpecialFunctions.$(fsym)))
                (!(is_special_func) || VERSION < v"0.6.0-dev.2767") && push!(tested_funcs, :(Base.$(fsym)))
                for func in tested_funcs
                    @eval begin
                        fdnum = $(is_domain_err_func ? FDNUM + 1 : FDNUM)
                        $(v) = ForwardDiff.value(fdnum)
                        @test duals_isapprox($(func)(fdnum), ForwardDiff.Dual($(func)($v), $(deriv) * ForwardDiff.partials(fdnum)))
                        if $(!(is_unsupported_nested_func))
                            nested_fdnum = $(is_domain_err_func ? NESTED_FDNUM + 1 : NESTED_FDNUM)
                            $(v) = ForwardDiff.value(nested_fdnum)
                            @test duals_isapprox($(func)(nested_fdnum), ForwardDiff.Dual($(func)($v), $(deriv) * ForwardDiff.partials(nested_fdnum)))
                        end
                    end
                end
            catch err
                warn("Encountered error when testing $(fsym)(::Dual):")
                rethrow(err)
            end
        end
    end

    # Special Cases #
    #---------------#

    @test dual_isapprox(hypot(FDNUM, FDNUM2), sqrt(FDNUM^2 + FDNUM2^2))
    @test dual_isapprox(hypot(FDNUM, FDNUM2, FDNUM), sqrt(2*(FDNUM^2) + FDNUM2^2))
    @test all(map(dual_isapprox, ForwardDiff.sincos(FDNUM), (sin(FDNUM), cos(FDNUM))))

    if V === Float32
        @test typeof(sqrt(FDNUM)) === typeof(FDNUM)
        @test typeof(sqrt(NESTED_FDNUM)) === typeof(NESTED_FDNUM)
    end
end

end # module
