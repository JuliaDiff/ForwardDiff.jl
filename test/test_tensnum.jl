using Base.Test
using Calculus
using ForwardDiff
using ForwardDiff: 
        GradientNum,
        HessianNum,
        TensorNum,
        value,
        grad,
        hess,
        tens,
        npartials,
        isconstant,
        hessnum,
        t_inds_2_h_ind

floatrange = .01:.01:.99
intrange = 1:10
N = 4
T = Float64
C = NTuple{N,T}
hessveclen = ForwardDiff.halfhesslen(N)
tensveclen = ForwardDiff.halftenslen(N)

test_val = rand(floatrange)
test_partials = tuple(rand(floatrange, N)...)
test_hessvec = rand(floatrange, hessveclen)
test_tensvec = rand(floatrange, tensveclen)

test_grad = GradientNum(test_val, test_partials)
test_hess = HessianNum(test_grad, test_hessvec)
test_tens = TensorNum(test_hess, test_tensvec)

######################
# Accessor Functions #
######################
@test value(test_tens) == test_val
@test grad(test_tens) == test_partials
@test hess(test_tens) == test_hessvec
@test tens(test_tens) == test_tensvec

for i in 1:N
    @test grad(test_tens, i) == test_partials[i]
end

for i in 1:hessveclen
    @test hess(test_tens, i) == test_hessvec[i]
end

for i in 1:tensveclen
    @test tens(test_tens, i) == test_tensvec[i]
end

@test npartials(test_tens) == npartials(typeof(test_tens)) == N

##################################
# Value Representation Functions #
##################################
@test eps(test_tens) == eps(test_val)
@test eps(typeof(test_tens)) == eps(T)

tens_zero = TensorNum(zero(test_hess), map(zero, test_tensvec))
tens_one = TensorNum(one(test_hess), map(zero, test_tensvec))

@test zero(test_tens) == tens_zero
@test zero(typeof(test_tens)) == tens_zero

@test one(test_tens) == tens_one
@test one(typeof(test_tens)) == tens_one

#########################################
# Conversion/Promotion/Hashing/Equality #
#########################################
int_val = round(Int, test_val)
int_partials = map(x -> round(Int, x), test_partials)
int_hessvec = map(x -> round(Int, x), test_hessvec)
int_tensvec = map(x -> round(Int, x), test_tensvec)

float_val = float(int_val)
float_partials = map(float, int_partials)
float_hessvec= map(float, int_hessvec)
float_tensvec= map(float, int_tensvec)

int_tens = TensorNum(HessianNum(GradientNum(int_val, int_partials), int_hessvec), int_tensvec)
float_tens = TensorNum(HessianNum(GradientNum(float_val, float_partials), float_hessvec), float_tensvec)
const_tens = TensorNum(float_val)

@test convert(typeof(test_tens), test_tens) == test_tens
@test convert(TensorNum, test_tens) == test_tens
@test convert(TensorNum{N,T,C}, int_tens) == float_tens
@test convert(TensorNum{0,T,Tuple{}}, 1) == TensorNum(1.0)
@test convert(TensorNum{3,T,NTuple{3,T}}, 1) == TensorNum{3,T,NTuple{3,T}}(1.0)
@test convert(T, TensorNum(HessianNum(GradientNum(1, tuple(0, 0))))) == 1.0

IntTens = TensorNum{N,Int,NTuple{N,Int}}
FloatTens = TensorNum{N,Float64,NTuple{N,Float64}}

@test promote_type(IntTens, IntTens) == IntTens
@test promote_type(FloatTens, IntTens) == FloatTens
@test promote_type(IntTens, Float64) == FloatTens
@test promote_type(FloatTens, Int) == FloatTens

@test hash(int_tens) == hash(float_tens)
@test hash(const_tens) == hash(float_val)

@test int_tens == float_tens
@test float_val == const_tens
@test const_tens == float_val

@test isequal(int_tens, float_tens)
@test isequal(float_val, const_tens)
@test isequal(const_tens, float_val)

@test copy(test_tens) == test_tens

####################
# is____ Functions #
####################
@test isnan(test_tens) == isnan(test_val)
@test isnan(TensorNum(NaN))

not_const_tens = TensorNum(HessianNum(GradientNum(one(T), map(one, test_partials))))
@test !(isconstant(not_const_tens) || isreal(not_const_tens))
@test isconstant(const_tens) && isreal(const_tens)
@test isconstant(zero(not_const_tens)) && isreal(zero(not_const_tens))

@test isfinite(test_tens) == isfinite(test_val)
@test !isfinite(TensorNum(Inf))

@test isless(test_tens-1, test_tens)
@test isless(test_val-1, test_tens)
@test isless(test_tens, test_val+1)

#######
# I/O #
#######
io = IOBuffer()
write(io, test_tens)
seekstart(io)

@test read(io, typeof(test_tens)) == test_tens

close(io)

##############
# Math tests #
##############
rand_val = rand(floatrange)
rand_partials = map(x -> rand(floatrange), test_partials)
rand_hessvec = map(x -> rand(floatrange), test_hessvec)
rand_tensvec = map(x -> rand(floatrange), test_tensvec)

rand_grad = GradientNum(rand_val, rand_partials)
rand_hess = HessianNum(rand_grad, rand_hessvec)
rand_tens = TensorNum(rand_hess, rand_tensvec)

# Addition/Subtraction #
#----------------------#
@test rand_tens + test_tens == TensorNum(rand_hess + test_hess, rand_tensvec + test_tensvec)
@test rand_tens + test_tens == test_tens + rand_tens
@test rand_tens - test_tens == TensorNum(rand_hess - test_hess, rand_tensvec - test_tensvec)

@test rand_val + test_tens == TensorNum(rand_val + test_hess, test_tensvec)
@test rand_val + test_tens == test_tens + rand_val
@test rand_val - test_tens == TensorNum(rand_val - test_hess, -test_tensvec)
@test test_tens - rand_val == TensorNum(test_hess - rand_val, test_tensvec)

@test -test_tens == TensorNum(-test_hess, -test_tensvec)

# Multiplication/Division #
#-------------------------#
function tens_approx_eq(a::TensorNum, b::TensorNum)
    eps = 1e-9
    try
        @test_approx_eq_eps value(a) value(b) eps
        @test_approx_eq_eps collect(grad(a)) collect(grad(b)) eps
        @test_approx_eq_eps hess(a) hess(b) eps
        @test_approx_eq_eps tens(a) tens(b) eps
    catch err
        error("Failure: TensorNum a and TensorNum b should be equal.\n Error: $err\n rand_tens = $rand_tens\n test_tens = $test_tens")
    end
end

@test hessnum(rand_tens * test_tens) == rand_hess * test_hess

@test rand_val * test_tens == TensorNum(rand_val * test_hess, rand_val * test_tensvec)
@test test_tens * rand_val == rand_val * test_tens

@test test_tens * true == test_tens
@test true * test_tens == test_tens * true
@test test_tens * false == zero(test_tens)
@test false * test_tens == test_tens * false

tens_approx_eq(rand_tens, rand_tens * (test_tens/test_tens))
tens_approx_eq(rand_tens, test_tens * (rand_tens/test_tens))
tens_approx_eq(rand_tens, (rand_tens * test_tens)/test_tens)
tens_approx_eq(test_tens, (rand_tens * test_tens)/rand_tens)

tens_approx_eq(2 * test_tens, test_tens + test_tens)
tens_approx_eq(test_tens * inv(test_tens), one(test_tens))

tens_approx_eq(rand_val / test_tens, rand_val * inv(test_tens))
tens_approx_eq(rand_val / test_tens, rand_val * 1/test_tens)
tens_approx_eq(rand_tens / test_tens, rand_tens * inv(test_tens))
tens_approx_eq(rand_tens / test_tens, rand_tens * 1/test_tens)
tens_approx_eq(test_tens / test_tens, one(test_tens))

@test test_tens / rand_val == TensorNum(test_hess / rand_val, test_tensvec / rand_val)

# Exponentiation #
#----------------#
tens_approx_eq(test_tens^rand_tens, exp(rand_tens * log(test_tens)))
tens_approx_eq(test_tens^rand_val, exp(rand_val * log(test_tens)))
tens_approx_eq(rand_val^test_tens, exp(test_tens * log(rand_val)))

# Univariate functions/API usage testing #
#----------------------------------------#
testout = Array(Float64, N, N, N)

function tens_deriv_ijk(f_expr, x::Vector, i, j, k)
    var_syms = [:a, :b, :c, :d]
    diff_expr = differentiate(f_expr, var_syms[k])
    diff_expr = differentiate(diff_expr, var_syms[j])
    diff_expr = differentiate(diff_expr, var_syms[i])
    @eval begin
        a,b,c,d = $x
        return $diff_expr
    end
end

function tens_test_result(f_expr, x::Vector)
    return [tens_deriv_ijk(f_expr, x, i, j, k) for i in 1:N, j in 1:N, k in 1:N]
end

function tens_test_x(fsym, N)
    randrange = .01:.01:.99

    needs_modification = tuple(:acosh, :acoth)
    if fsym in needs_modification
        randrange += 1
    end

    return rand(randrange, N)
end

for fsym in ForwardDiff.univar_tens_funcs
    try 
        testexpr = :($(fsym)(a) + $(fsym)(b) - $(fsym)(c) * $(fsym)(d))

        @eval function testf(x::Vector) 
            a,b,c,d = x
            return $testexpr
        end

        testx = tens_test_x(fsym, N)
        testresult = tens_test_result(testexpr, testx)

        ForwardDiff.tensor!(testout, testf, testx)
        @test_approx_eq testout testresult

        @test_approx_eq ForwardDiff.tensor(testf, testx) testresult

        tensf! = ForwardDiff.tensor(testf, mutates=true)
        tensf!(testout, testx)
        @test_approx_eq testout testresult

        tensf = ForwardDiff.tensor(testf, mutates=false)
        @test_approx_eq tensf(testx) testresult
    catch err
        error("Failure when testing Tensors involving $fsym: $err")
    end
end
