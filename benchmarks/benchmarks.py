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
def ackley_term(a,b):
    return (-20 * algopy.exp(-0.2*algopy.sqrt(0.5*(a**2 + b**2))) - 
            algopy.exp(0.5*(algopy.cos(2*math.pi*a) + 
            algopy.cos(2*math.pi*b))) + math.e + 20)

def ackley_sum(x):
    return sum(ackley_term(x[i], x[i+1]) for i in range(len(x)-1))

#############################
# Benchmark utility methods #
#############################
# Usage:
#
# # The values of the dict are arrays of time values, where indices correspond to len(x)
# In [16]: t = bench_fad(ackley_sum, range(10,100,10), 4) # benchmark ackley_sum where len(x) = range(10,100,10), taking the minimum of 4 trials
# In [17]: t
# Out[17]:
# {'ftimes': [0.00017690658569335938,
#   0.00036406517028808594,
#   0.0005707740783691406,
#   0.0007112026214599609,
#   0.0007939338684082031,
#   0.000965118408203125,
#   0.0011258125305175781,
#   0.0012819766998291016,
#   0.001363992691040039],
#  'gtimes': [0.003047943115234375,
#   0.007333040237426758,
#   0.009442806243896484,
#   0.012690067291259766,
#   0.016045808792114258,
#   0.019411087036132812,
#   0.023248910903930664,
#   0.026779890060424805,
#   0.03076004981994629],
#  'htimes': [0.0044209957122802734,
#   0.010719060897827148,
#   0.01880502700805664,
#   0.03013896942138672,
#   0.04469108581542969,
#   0.06389999389648438,
#   0.08915901184082031,
#   0.12110686302185059,
#   0.1636369228363037]}

def bench_fad(f, itr, repeat):
    fname = f.__name__
    import_stmt = 'import numpy, algopy, math; from __main__ import ' + fname + ', gradient, hessian;'
    return {'ftimes': bench_range(fname + '(x)', import_stmt, itr, repeat),
            'gtimes': bench_range('g(x)', import_stmt + 'g = gradient(' + fname + ');', itr, repeat), 
            'htimes': bench_range('h(x)', import_stmt + 'h = hessian(' + fname + ');', itr, repeat)}

def bench_range(stmt, setup, itr, repeat):
    x_stmt = lambda xlen: 'x = numpy.random.rand(' + str(xlen) + ')'
    return [min(timeit.repeat(stmt, setup=(setup + x_stmt(i)), number=1, repeat=repeat)) for i in itr]

