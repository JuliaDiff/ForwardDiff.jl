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
    sum_sqrs = sum(i**2 for i in x)
    sum_cos = sum(algopy.cos(2*math.pi*i) for i in x)
    len_recip = 1/len(x)
    return (-20 * algopy.exp(-0.2*algopy.sqrt(len_recip*sum_sqrs)) - 
            algopy.exp(0.5*sum_cos) + 20 + math.e)

#############################
# Benchmark utility methods #
#############################
# Usage:
#
# # The values of the dict are arrays of time values, where indices correspond to len(x)
# In [11]: t = bench_fad(ackley, range(10,100,10), 4) # benchmark ackley where len(x) = range(10,100,10), taking the minimum of 4 trials
# In [12]: t
# Out[12]:
# {'ftimes': [5.507469177246094e-05,
#   9.799003601074219e-05,
#   0.00013899803161621094,
#   0.00017786026000976562,
#   0.00020694732666015625,
#   0.0002589225769042969,
#   0.0002429485321044922,
#   0.00027489662170410156,
#   0.0003108978271484375],
#  'gtimes': [0.0013709068298339844,
#   0.002513885498046875,
#   0.0037069320678710938,
#   0.0048830509185791016,
#   0.0060579776763916016,
#   0.007241010665893555,
#   0.008578062057495117,
#   0.009813070297241211,
#   0.011088132858276367],
#  'htimes': [0.0021181106567382812,
#   0.00468897819519043,
#   0.008304834365844727,
#   0.013358831405639648,
#   0.021745920181274414,
#   0.028542041778564453,
#   0.04112100601196289,
#   0.055689096450805664,
#   0.07504796981811523]}
#
# In [14]: t = bench_fad(ackley, (400,), 4)
# In [15]: t
# Out[15]:
# {'ftimes': [0.0014510154724121094],
#  'gtimes': [0.060678958892822266],
#  'htimes': [10.761451959609985]}

def bench_fad(f, itr, repeat):
    fname = f.__name__
    import_stmt = 'import numpy, algopy, math; from __main__ import ' + fname + ', gradient, hessian;'
    return {'ftimes': bench_range(fname + '(x)', import_stmt, itr, repeat),
            'gtimes': bench_range('g(x)', import_stmt + 'g = gradient(' + fname + ');', itr, repeat), 
            'htimes': bench_range('h(x)', import_stmt + 'h = hessian(' + fname + ');', itr, repeat)}

def bench_range(stmt, setup, itr, repeat):
    x_stmt = lambda xlen: 'x = numpy.random.rand(' + str(xlen) + ')'
    return [min(timeit.repeat(stmt, setup=(setup + x_stmt(i)), number=1, repeat=repeat)) for i in itr]

