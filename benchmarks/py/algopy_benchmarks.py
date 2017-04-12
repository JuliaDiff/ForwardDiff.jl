import numpy, algopy, timeit #, os, pickle

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
    a, b, c = 20.0, -0.2, 2.0*numpy.pi
    len_recip = 1.0/len(x)
    sum_sqrs, sum_cos = 0.0, 0.0
    for i in x:
        sum_cos += algopy.cos(c*i)
        sum_sqrs += i*i
    return (-a * algopy.exp(b*algopy.sqrt(len_recip*sum_sqrs)) -
            algopy.exp(len_recip*sum_cos) + a + numpy.e)

def rosenbrock(x):
    a, b = 100.0, 1.0
    result = 0.0
    for i in xrange(len(x)-1):
        result += (b - x[i])**2 + a*(x[i+1] - x[i]**2)**2
    return result

def self_weighted_logit(x):
    return 1.0/(1.0 + algopy.exp(-algopy.dot(x, x)))

#############################
# Benchmark utility methods #
#############################

def bench(f, x, repeat):
    def wrapf():
        return f(x)
    return min(timeit.repeat(wrapf, number=1, repeat=repeat))
