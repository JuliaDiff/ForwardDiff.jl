using Base.Test
using ForwardDiff
using Calculus

floatrange = 0.0:.01:.99
intrange = 0:10
N = 3
T = Float64

testval = rand(floatrange)
partialtup = tuple(rand(floatrange, N)...)
partialvec = collect(partialtup)

for (testpartials, Grad) in ((partialtup, ForwardDiff.GradNumTup), (partialvec, ForwardDiff.GradNumVec))
    testgrad = Grad{N,T}(testval, testpartials)

    ######################
    # Accessor Functions #
    ######################
    @test ForwardDiff.value(testgrad) == testval
    @test ForwardDiff.grad(testgrad) == testpartials

    for i in 1:N
        @test ForwardDiff.grad(testgrad, i) == testpartials[i]
    end

    @test ForwardDiff.npartials(testgrad) == ForwardDiff.npartials(typeof(testgrad)) == N
    
    ##################################
    # Value Representation Functions #
    ##################################
    @test eps(testgrad) == eps(testval)
    @test eps(typeof(testgrad)) == eps(T)

    grad_zero = Grad{N,T}(zero(testval), map(zero, testpartials))
    grad_one = Grad{N,T}(one(testval), map(zero, testpartials))

    @test zero(testgrad) == grad_zero
    @test zero(typeof(testgrad)) == grad_zero

    @test one(testgrad) == grad_one
    @test one(typeof(testgrad)) == grad_one

    #########################################
    # Conversion/Promotion/Hashing/Equality #
    #########################################
    int_val = round(Int, testval)
    int_derivs = map(x -> round(Int, x), testpartials)
    float_val = float(int_val)
    float_derivs = map(float, int_derivs)

    int_grad = Grad{N,Int}(int_val, int_derivs)
    float_grad = Grad{N,Float64}(float_val, float_derivs)
    const_grad = Grad{N,Float64}(float_val)

    @test convert(typeof(testgrad), testgrad) == testgrad
    @test convert(ForwardDiff.GradientNum, testgrad) == testgrad
    @test convert(Grad{N,Float64}, int_grad) == float_grad
    @test convert(Grad{0,Float64}, 1) == Grad{0,Float64}(1.0, ForwardDiff.zero_partials(Grad{0,Float64}))
    @test convert(Grad{3,Float64}, 1) == Grad{3,Float64}(1.0, ForwardDiff.zero_partials(Grad{3,Float64}))
    @test convert(Float64, Grad{2,Float64}(1)) == 1.0

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

    @test copy(testgrad) == testgrad

    ####################
    # is____ Functions #
    ####################
    @test isnan(testgrad) == isnan(testval)
    @test isnan(Grad{0,Float64}(NaN))

    not_const_grad = Grad{N,T}(1, map(one, testpartials))
    @test !(ForwardDiff.isconstant(not_const_grad) || isreal(not_const_grad) || any(x -> x == 0, ForwardDiff.grad(not_const_grad)))
    @test ForwardDiff.isconstant(const_grad) && isreal(const_grad)
    @test ForwardDiff.isconstant(zero(not_const_grad)) && isreal(zero(not_const_grad))

    @test isfinite(testgrad) && isfinite(testval)
    @test !isfinite(Grad{N,T}(Inf))

    @test isless(testgrad-1, testgrad)
    @test isless(testval-1, testgrad)
    @test isless(testgrad, testval+1)

    #######
    # I/O #
    #######
    io = IOBuffer()
    write(io, testgrad)
    seekstart(io)

    @test read(io, typeof(testgrad)) == testgrad
    
    close(io)

    #####################################
    # Arithmetic/Mathematical Functions #
    #####################################
    randval = rand(T)
    randpartials = map(x -> rand(T), testpartials)
    randgrad = Grad{N,T}(randval, randpartials)

    # Addition/Subtraction #
    #----------------------#
    @test randgrad + testgrad == Grad{N,T}(randval+testval, map(+, randpartials, testpartials))
    @test randgrad + testgrad == testgrad + randgrad
    @test randgrad - testgrad == Grad{N,T}(randval-testval, map(-, randpartials, testpartials))
    @test randval - testgrad == Grad{N,T}(randval-testval, map(-, testpartials))
    @test testgrad - randval == Grad{N,T}(testval-randval, testpartials)
    @test -testgrad == Grad{N,T}(-testval, map(-, testpartials))

    # Multiplication #
    #----------------#
    randxtest = randgrad * testgrad

    @test ForwardDiff.value(randxtest) == randval * testval

    for i in 1:N
        @test ForwardDiff.grad(randxtest, i) == (randpartials[i] * testval) + (randval * testpartials[i])
    end

    @test randval * testgrad == Grad{N,T}(randval*testval, map(x -> randval*x, testpartials))
    @test testgrad * randval == randval * testgrad

    @test testgrad * false == zero(testgrad)
    @test testgrad * true == testgrad
    @test true * testgrad == testgrad * true
    @test false * testgrad == testgrad * false

    # Division #
    #----------#
    randdivtest = randgrad / testgrad
    valdivtest = randval / testgrad

    @test ForwardDiff.value(randdivtest) == randval/testval
    @test ForwardDiff.value(valdivtest) == randval/testval

    for i in 1:N
        @test ForwardDiff.grad(randdivtest, i) == ((randpartials[i] * testval) - (randval * testpartials[i])) / testval^2
        @test ForwardDiff.grad(valdivtest, i) == (-randval * testpartials[i]) / testval^2
    end

    @test testgrad / randval == Grad{N,T}(testval/randval, map(x -> x/randval, testpartials))

    # Exponentiation #
    #----------------#
    powval = randval * (testval^(randval-1))
    logval = (testval^randval) * log(testval)

    testexprand = testgrad^randgrad
    valexptest = randval^testgrad
    testexpval = testgrad^randval

    @test ForwardDiff.value(testexprand) == testval^randval
    @test ForwardDiff.value(valexptest) == randval^testval
    @test ForwardDiff.value(testexpval) == testval^randval

    for i in 1:N
        @test_approx_eq ForwardDiff.grad(testexprand, i) (testpartials[i] * powval) + (logval * randpartials[i])
        @test_approx_eq ForwardDiff.grad(valexptest, i) testpartials[i] * ((randval^testval) * log(randval))
        @test_approx_eq ForwardDiff.grad(testexpval, i) testpartials[i] * randval * (testval^(randval-1))

    end

    # Univariate functions #
    #----------------------#
    @test abs(testgrad) == testgrad
    @test abs(-testgrad) == testgrad
    @test abs2(testgrad) == testgrad*testgrad

    @test conj(testgrad) == testgrad
    @test transpose(testgrad) == testgrad
    @test ctranspose(testgrad) == testgrad

    for (fsym, expr) in Calculus.symbolic_derivatives_1arg()
        @eval begin
            func = $fsym

            local orig_grad, fgrad

            try
                orig_grad = $testgrad
                fgrad = func(orig_grad)
            catch DomainError
                # some of the provided functions
                # have a domain x > 1, so we simply
                # add 1 to our test GradientNum if
                # a DomainError is thrown
                orig_grad = $testgrad + 1
                fgrad = func(orig_grad)
            end

            x = ForwardDiff.value(orig_grad)
            df = $expr 

            @test_approx_eq ForwardDiff.value(fgrad) func(x)

            for i in 1:N
                try 
                    @test_approx_eq ForwardDiff.grad(fgrad, i) df*ForwardDiff.grad(orig_grad, i)
                catch exception
                    info("The exception was thrown while testing function $func at value $orig_grad")
                    throw(exception)
                end
            end
        end
    end
end
