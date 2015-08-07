using Base.Test
using Calculus
using ForwardDiff
using ForwardDiff: 
        GradientNum,
        value,
        grad,
        npartials,
        isconstant

floatrange = 0.0:.01:.99
intrange = 0:10
N = 3
T = Float64

test_val = rand(floatrange)
test_partialstup = tuple(rand(floatrange, N)...)
test_partialsvec = collect(test_partialstup)

for (test_partials, Grad) in ((test_partialstup, ForwardDiff.GradNumTup), (test_partialsvec, ForwardDiff.GradNumVec))
    test_grad = Grad{N,T}(test_val, test_partials)

    ######################
    # Accessor Functions #
    ######################
    @test value(test_grad) == test_val
    @test grad(test_grad) == test_partials

    for i in 1:N
        @test grad(test_grad, i) == test_partials[i]
    end

    @test npartials(test_grad) == npartials(typeof(test_grad)) == N
    
    ##################################
    # Value Representation Functions #
    ##################################
    @test eps(test_grad) == eps(test_val)
    @test eps(typeof(test_grad)) == eps(T)

    grad_zero = Grad{N,T}(zero(test_val), map(zero, test_partials))
    grad_one = Grad{N,T}(one(test_val), map(zero, test_partials))

    @test zero(test_grad) == grad_zero
    @test zero(typeof(test_grad)) == grad_zero

    @test one(test_grad) == grad_one
    @test one(typeof(test_grad)) == grad_one

    #########################################
    # Conversion/Promotion/Hashing/Equality #
    #########################################
    int_val = round(Int, test_val)
    int_partials = map(x -> round(Int, x), test_partials)
    float_val = float(int_val)
    float_partials = map(float, int_partials)

    int_grad = Grad{N,Int}(int_val, int_partials)
    float_grad = Grad{N,T}(float_val, float_partials)
    const_grad = Grad{N,T}(float_val)

    @test convert(typeof(test_grad), test_grad) == test_grad
    @test convert(GradientNum, test_grad) == test_grad
    @test convert(Grad{N,T}, int_grad) == float_grad
    @test convert(Grad{0,T}, 1) == Grad{0,T}(1.0, ForwardDiff.zero_partials(Grad{0,T}))
    @test convert(Grad{3,T}, 1) == Grad{3,T}(1.0, ForwardDiff.zero_partials(Grad{3,T}))
    @test convert(T, Grad{2,T}(1)) == 1.0

    @test promote_type(Grad{N,Int}, Grad{N,Int}) == Grad{N,Int}
    @test promote_type(Grad{N,Float64}, Grad{N,Int}) == Grad{N,Float64}
    @test promote_type(Grad{N,Int}, Float64) == Grad{N,Float64}
    @test promote_type(Grad{N,Float64}, Int) == Grad{N,Float64}

    @test hash(int_grad) == hash(float_grad)
    @test hash(const_grad) == hash(float_val)

    @test int_grad == float_grad
    @test float_val == const_grad
    @test const_grad == float_val

    @test isequal(int_grad, float_grad)
    @test isequal(float_val, const_grad)
    @test isequal(const_grad, float_val)

    @test copy(test_grad) == test_grad

    ####################
    # is____ Functions #
    ####################
    @test isnan(test_grad) == isnan(test_val)
    @test isnan(Grad{0,T}(NaN))

    not_const_grad = Grad{N,T}(1, map(one, test_partials))
    @test !(isconstant(not_const_grad) || isreal(not_const_grad))
    @test isconstant(const_grad) && isreal(const_grad)
    @test isconstant(zero(not_const_grad)) && isreal(zero(not_const_grad))

    @test isfinite(test_grad) && isfinite(test_val)
    @test !isfinite(Grad{N,T}(Inf))

    @test isless(test_grad-1, test_grad)
    @test isless(test_val-1, test_grad)
    @test isless(test_grad, test_val+1)

    #######
    # I/O #
    #######
    io = IOBuffer()
    write(io, test_grad)
    seekstart(io)

    @test read(io, typeof(test_grad)) == test_grad
    
    close(io)

    #####################################
    # Arithmetic/Mathematical Functions #
    #####################################
    rand_val = rand(T)
    rand_partials = map(x -> rand(T), test_partials)
    rand_grad = Grad{N,T}(rand_val, rand_partials)

    # Addition/Subtraction #
    #----------------------#
    @test rand_grad + test_grad == Grad{N,T}(rand_val+test_val, map(+, rand_partials, test_partials))
    @test rand_grad + test_grad == test_grad + rand_grad
    @test rand_grad - test_grad == Grad{N,T}(rand_val-test_val, map(-, rand_partials, test_partials))

    @test rand_val + test_grad == Grad{N,T}(rand_val+test_val, test_partials)
    @test rand_val + test_grad == test_grad + rand_val
    @test rand_val - test_grad == Grad{N,T}(rand_val-test_val, map(-, test_partials))
    @test test_grad - rand_val == Grad{N,T}(test_val-rand_val, test_partials)
    
    @test -test_grad == Grad{N,T}(-test_val, map(-, test_partials))

    # Multiplication #
    #----------------#
    rand_x_test = rand_grad * test_grad

    @test value(rand_x_test) == rand_val * test_val

    for i in 1:N
        @test grad(rand_x_test, i) == (rand_partials[i] * test_val) + (rand_val * test_partials[i])
    end

    @test rand_val * test_grad == Grad{N,T}(rand_val*test_val, map(x -> rand_val*x, test_partials))
    @test test_grad * rand_val == rand_val * test_grad

    @test test_grad * false == zero(test_grad)
    @test test_grad * true == test_grad
    @test true * test_grad == test_grad * true
    @test false * test_grad == test_grad * false

    # Division #
    #----------#
    function grad_approx_eq(a::GradientNum, b::GradientNum)
        @test_approx_eq value(a) value(b)
        @test_approx_eq collect(grad(a)) collect(grad(b))
    end

    grad_approx_eq(rand_grad / test_grad, rand_grad * inv(test_grad))
    grad_approx_eq(rand_val / test_grad, rand_val * inv(test_grad))

    @test test_grad / rand_val == Grad{N,T}(test_val/rand_val, map(x -> x/rand_val, test_partials))

    # Exponentiation #
    #----------------#
    grad_approx_eq(test_grad^rand_grad, exp(rand_grad * log(test_grad)))
    grad_approx_eq(test_grad^rand_val, exp(rand_val * log(test_grad)))
    grad_approx_eq(rand_val^test_grad, exp(test_grad * log(rand_val)))

    # Univariate functions #
    #----------------------#
    @test abs(test_grad) == test_grad
    @test abs(-test_grad) == test_grad
    @test abs2(test_grad) == test_grad*test_grad

    @test conj(test_grad) == test_grad
    @test transpose(test_grad) == test_grad
    @test ctranspose(test_grad) == test_grad

    for (fsym, expr) in Calculus.symbolic_derivatives_1arg()
        @eval begin
            func = $fsym

            local orig_grad, f_grad

            try
                orig_grad = $test_grad
                f_grad = func(orig_grad)
            catch DomainError
                # some of the provided functions
                # have a domain x > 1, so we simply
                # add 1 to our test GradientNum if
                # a DomainError is thrown
                orig_grad = $test_grad + 1
                f_grad = func(orig_grad)
            end

            x = value(orig_grad)
            df = $expr 

            @test_approx_eq value(f_grad) func(x)

            for i in 1:N
                try 
                    @test_approx_eq grad(f_grad, i) df*grad(orig_grad, i)
                catch exception
                    info("The exception was thrown while testing function $func at value $orig_grad")
                    throw(exception)
                end
            end
        end
    end
end

#####################
# API Usage Testing #
#####################
N = 4
testout = Array(Float64, N)

function grad_deriv_i(f_expr, x::Vector, i)
    var_syms = [:a, :b, :c, :d]
    diff_expr = differentiate(f_expr, var_syms[i])
    @eval begin
        a,b,c,d = $x
        return $diff_expr
    end
end

function grad_test_result(f_expr, x::Vector)
    return [grad_deriv_i(f_expr, x, i) for i in 1:N]
end

function grad_test_x(fsym, N)
    randrange = 0.01:.01:.99

    needs_modification = (:asec, :acsc, :asecd, :acscd, :acosh, :acoth)
    if fsym in needs_modification
        randrange += 1
    end

    return rand(randrange, N)
end

for fsym in map(first, Calculus.symbolic_derivatives_1arg())
    try
        testexpr = :($(fsym)(a) + $(fsym)(b) - $(fsym)(c) * $(fsym)(d)) 

        @eval function testf(x::Vector) 
            a,b,c,d = x
            return $testexpr
        end

        testx = grad_test_x(fsym, N)
        testresult = grad_test_result(testexpr, testx)

        ForwardDiff.gradient!(testout, testf, testx)
        @test_approx_eq testout testresult

        @test_approx_eq ForwardDiff.gradient(testf, testx) testresult

        gradf! = ForwardDiff.gradient(testf, mutates=true)
        gradf!(testout, testx)
        @test_approx_eq testout testresult

        gradf = ForwardDiff.gradient(testf, mutates=false)
        @test_approx_eq gradf(testx) testresult
    catch err
        error("Failure when testing gradients involving $fsym: $err")
    end
end
