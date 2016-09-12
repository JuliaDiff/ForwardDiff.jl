import timeit
import autograd.numpy as np
from autograd import grad

def ackley(x):
    a, b, c = 20.0, -0.2, 2.0*np.pi
    len_recip = 1.0/len(x)
    sum_sqrs = sum(x*x)
    sum_cos = sum(np.cos(c*x))
    return (-a * np.exp(b*np.sqrt(len_recip*sum_sqrs)) -
            np.exp(len_recip*sum_cos) + a + np.e)

def rosenbrock(x):
    a, b = 100.0, 1.0
    return sum((b - x[0:-1])**2 + a*(x[1:] - x[0:-1]**2)**2)

def self_weighted_logit(x):
    return 1.0/(1.0 + np.exp(-np.dot(x, x)))

def bench(f, x, r, n):
    def wrapf():
        return f(x)
    return min(t/n for t in timeit.repeat(wrapf, number=n, repeat=r))

# def devec_ackley(x):
#     a, b, c = 20.0, -0.2, 2.0*np.pi
#     len_recip = 1.0/len(x)
#     sum_sqrs, sum_cos = 0.0, 0.0
#     for i in x:
#         sum_cos += np.cos(c*i)
#         sum_sqrs += i*i
#     return (-a * np.exp(b*np.sqrt(len_recip*sum_sqrs)) -
#             np.exp(len_recip*sum_cos) + a + np.e)
#
# def devec_rosenbrock(x):
#     a, b = 100.0, 1.0
#     result = 0.0
#     for i in xrange(len(x)-1):
#         result += (b - x[i])**2 + a*(x[i+1] - x[i]**2)**2
#     return result
