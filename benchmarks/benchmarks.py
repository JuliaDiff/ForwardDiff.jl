import numpy, algopy, math, timeit

################
# AD functions #
################
def gradient(f):
    def gradf(x):
        y = algopy.UTPM.init_jacobian(x)
        return algopy.UTPM.extract_jacobian(f(y))
    return gradf

def hessian(f):
    def hessf(x):
        y = algopy.UTPM.init_hessian(x)
        return algopy.UTPM.extract_hessian(x.size, f(y))
    return hessf

##################
# Test functions #
##################
def ackley(x):
    a, b = 20.0, -0.2
    len_recip = 1.0/len(x)
    sum_sqrs, sum_cos = 0.0, 0.0
    for i in x:
        sum_sqrs += i**2
        sum_cos += algopy.cos(2.0*math.pi*i)
    return (-a * algopy.exp(b*algopy.sqrt(len_recip*sum_sqrs)) -
            algopy.exp(len_recip*sum_cos) + a + math.e)

#############################
# Benchmark utility methods #
#############################
# Usage:
#
# benchmark ackley where len(x) = range(10,100,10), taking the minimum of 4 trials:
# bench_fad(ackley, range(10,100,10), 4)
#
# benchmark ackley where len(x) = 400, taking the minimum of 8 trials:
# bench_fad(ackley, (400,), 8)

def bench_fad(f, itr, repeat):
    fname = f.__name__
    import_stmt = 'import numpy, algopy, math; from __main__ import ' + fname + ', gradient, hessian;'
    return {'ftimes': bench_range(fname + '(x)', import_stmt, itr, repeat),
            'gtimes': bench_range('g(x)', import_stmt + 'g = gradient(' + fname + ');', itr, repeat), 
            'htimes': bench_range('h(x)', import_stmt + 'h = hessian(' + fname + ');', itr, repeat)}

def bench_range(stmt, setup, itr, repeat):
    x_stmt = lambda xlen: 'x = numpy.random.rand(' + str(xlen) + ')'
    return [min(timeit.repeat(stmt, setup=(setup + x_stmt(i)), number=1, repeat=repeat)) for i in itr]

