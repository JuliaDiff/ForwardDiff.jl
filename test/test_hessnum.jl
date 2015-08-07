using Base.Test
using Calculus
using ForwardDiff
using ForwardDiff: 
        GradientNum,
        HessianNum,
        value,
        grad,
        hess,
        npartials,
        isconstant,
        gradnum

floatrange = .01:.01:.99
intrange = 1:10
N = 4
T = Float64
C = NTuple{N,T}
hessveclen = ForwardDiff.halfhesslen(N)

test_val = rand(floatrange)
test_partials = tuple(rand(floatrange, N)...)
test_hessvec = rand(floatrange, hessveclen)
test_grad = GradientNum(test_val, test_partials)
test_hess = HessianNum(test_grad, test_hessvec)

######################
# Accessor Functions #
######################
@test value(test_hess) == test_val
@test grad(test_hess) == test_partials
@test hess(test_hess) == test_hessvec

for i in 1:N
    @test grad(test_hess, i) == test_partials[i]
end

for i in 1:hessveclen
    @test hess(test_hess, i) == test_hessvec[i]
end

@test npartials(test_hess) == npartials(typeof(test_hess)) == N

##################################
# Value Representation Functions #
##################################
@test eps(test_hess) == eps(test_val)
@test eps(typeof(test_hess)) == eps(T)

hess_zero = HessianNum(zero(test_grad), map(zero, test_hessvec))
hess_one = HessianNum(one(test_grad), map(zero, test_hessvec))

@test zero(test_hess) == hess_zero
@test zero(typeof(test_hess)) == hess_zero

@test one(test_hess) == hess_one
@test one(typeof(test_hess)) == hess_one

#########################################
# Conversion/Promotion/Hashing/Equality #
#########################################
int_val = round(Int, test_val)
int_partials = map(x -> round(Int, x), test_partials)
int_hessvec = map(x -> round(Int, x), test_hessvec)

float_val = float(int_val)
float_partials = map(float, int_partials)
float_hessvec= map(float, int_hessvec)

int_hess = HessianNum(GradientNum(int_val, int_partials), int_hessvec)
float_hess = HessianNum(GradientNum(float_val, float_partials), float_hessvec)
const_hess = HessianNum(float_val)

@test convert(typeof(test_hess), test_hess) == test_hess
@test convert(HessianNum, test_hess) == test_hess
@test convert(HessianNum{N,T,C}, int_hess) == float_hess
@test convert(HessianNum{0,T,Tuple{}}, 1) == HessianNum(1.0)
@test convert(HessianNum{3,T,NTuple{3,T}}, 1) == HessianNum{3,T,NTuple{3,T}}(1.0)
@test convert(T, HessianNum(GradientNum(1, tuple(0, 0)))) == 1.0

IntHess = HessianNum{N,Int,NTuple{N,Int}}
FloatHess = HessianNum{N,Float64,NTuple{N,Float64}}

@test promote_type(IntHess, IntHess) == IntHess
@test promote_type(FloatHess, IntHess) == FloatHess
@test promote_type(IntHess, Float64) == FloatHess
@test promote_type(FloatHess, Int) == FloatHess

@test hash(int_hess) == hash(float_hess)
@test hash(const_hess) == hash(float_val)

@test int_hess == float_hess
@test float_val == const_hess
@test const_hess == float_val

@test isequal(int_hess, float_hess)
@test isequal(float_val, const_hess)
@test isequal(const_hess, float_val)

@test copy(test_hess) == test_hess

####################
# is____ Functions #
####################
@test isnan(test_hess) == isnan(test_val)
@test isnan(HessianNum(NaN))

not_const_hess = HessianNum(GradientNum(one(T), map(one, test_partials)))
@test !(isconstant(not_const_hess) || isreal(not_const_hess))
@test isconstant(const_hess) && isreal(const_hess)
@test isconstant(zero(not_const_hess)) && isreal(zero(not_const_hess))

@test isfinite(test_hess) == isfinite(test_val)
@test !isfinite(HessianNum(Inf))

@test isless(test_hess-1, test_hess)
@test isless(test_val-1, test_hess)
@test isless(test_hess, test_val+1)

#######
# I/O #
#######
io = IOBuffer()
write(io, test_hess)
seekstart(io)

@test read(io, typeof(test_hess)) == test_hess

close(io)

##############
# Math tests #
##############
rand_val = rand(floatrange)
rand_partials = map(x -> rand(floatrange), test_partials)
rand_hessvec = map(x -> rand(floatrange), test_hessvec)
rand_grad = GradientNum(rand_val, rand_partials)
rand_hess = HessianNum(rand_grad, rand_hessvec)

# Addition/Subtraction #
#----------------------#
@test rand_hess + test_hess == HessianNum(rand_grad + test_grad, rand_hessvec + test_hessvec)
@test rand_hess + test_hess == test_hess + rand_hess
@test rand_hess - test_hess == HessianNum(rand_grad - test_grad, rand_hessvec - test_hessvec)

@test rand_val + test_hess == HessianNum(rand_val + test_grad, test_hessvec)
@test rand_val + test_hess == test_hess + rand_val
@test rand_val - test_hess == HessianNum(rand_val - test_grad, -test_hessvec)
@test test_hess - rand_val == HessianNum(test_grad - rand_val, test_hessvec)

@test -test_hess == HessianNum(-test_grad, -test_hessvec)

# Multiplication/Division #
#-------------------------#
function hess_approx_eq(a::HessianNum, b::HessianNum)
    eps = 1e-9
    try
        @test_approx_eq_eps value(a) value(b) eps
        @test_approx_eq_eps collect(grad(a)) collect(grad(b)) eps
        @test_approx_eq_eps hess(a) hess(b) eps
    catch err
        error("Failure: HessianNum a and HessianNum b should be equal.\n Error: $err\n rand_hess = $rand_hess\n test_hess = $test_hess")
    end
end

@test gradnum(rand_hess * test_hess) == rand_grad * test_grad

@test rand_val * test_hess == HessianNum(rand_val * test_grad, rand_val * test_hessvec)
@test test_hess * rand_val == rand_val * test_hess

@test test_hess * true == test_hess
@test true * test_hess == test_hess * true
@test test_hess * false == zero(test_hess)
@test false * test_hess == test_hess * false

hess_approx_eq(rand_hess, rand_hess * (test_hess/test_hess))
hess_approx_eq(rand_hess, test_hess * (rand_hess/test_hess))
hess_approx_eq(rand_hess, (rand_hess * test_hess)/test_hess)
hess_approx_eq(test_hess, (rand_hess * test_hess)/rand_hess)

hess_approx_eq(2 * test_hess, test_hess + test_hess)
hess_approx_eq(test_hess * inv(test_hess), one(test_hess))

hess_approx_eq(rand_val / test_hess, rand_val * inv(test_hess))
hess_approx_eq(rand_val / test_hess, rand_val * 1/test_hess)
hess_approx_eq(rand_hess / test_hess, rand_hess * inv(test_hess))
hess_approx_eq(rand_hess / test_hess, rand_hess * 1/test_hess)
hess_approx_eq(test_hess / test_hess, one(test_hess))

@test test_hess / rand_val == HessianNum(test_grad / rand_val, test_hessvec / rand_val)

# Exponentiation #
#----------------#
hess_approx_eq(test_hess^rand_hess, exp(rand_hess * log(test_hess)))
hess_approx_eq(test_hess^rand_val, exp(rand_val * log(test_hess)))
hess_approx_eq(rand_val^test_hess, exp(test_hess * log(rand_val)))

# Univariate functions/API usage testing #
#----------------------------------------#
testout = Array(Float64, N, N)

function hess_deriv_ij(f_expr, x::Vector, i, j)
    var_syms = [:a, :b, :c, :d]
    diff_expr = differentiate(f_expr, var_syms[j])
    diff_expr = differentiate(diff_expr, var_syms[i])
    @eval begin
        a,b,c,d = $x
        return $diff_expr
    end
end

function hess_test_result(f_expr, x::Vector)
    return [hess_deriv_ij(f_expr, x, i, j) for i in 1:N, j in 1:N]
end

function hess_test_x(fsym, N)
    randrange = .01:.01:.99

    needs_modification = (:acosh, :acoth)
    if fsym in needs_modification
        randrange += 1
    end

    return rand(randrange, N)
end

for fsym in ForwardDiff.univar_hess_funcs
    try    
        testexpr = :($(fsym)(a) + $(fsym)(b) - $(fsym)(c) * $(fsym)(d)) 

        @eval function testf(x::Vector) 
            a,b,c,d = x
            return $testexpr
        end

        testx = hess_test_x(fsym, N)
        testresult = hess_test_result(testexpr, testx)

        ForwardDiff.hessian!(testout, testf, testx)
        @test_approx_eq testout testresult

        @test_approx_eq ForwardDiff.hessian(testf, testx) testresult

        hessf! = ForwardDiff.hessian(testf, mutates=true)
        hessf!(testout, testx)
        @test_approx_eq testout testresult

        hessf = ForwardDiff.hessian(testf, mutates=false)
        @test_approx_eq hessf(testx) testresult
    catch err
        error("Failure when testing Hessians involving $fsym: $err")
    end
end