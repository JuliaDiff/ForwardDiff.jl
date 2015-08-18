import numpy, algopy, timeit, os, pickle

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
def sqr(i):
    return i*i

def ackley(x):
    a, b, c = 20.0, -0.2, 2.0*numpy.pi
    len_recip = 1.0/len(x)
    sum_sqrs, sum_cos = 0.0, 0.0
    for i in x:
        sum_cos += algopy.cos(c*i)
        sum_sqrs += sqr(i)
    return (-a * algopy.exp(b*algopy.sqrt(len_recip*sum_sqrs)) -
            algopy.exp(len_recip*sum_cos) + a + numpy.e)

def rosenbrock(x):
    a, b = 100.0, 1.0
    result = 0.0
    for i in xrange(len(x)-1):
        result += sqr(b - x[i]) + a*sqr(x[i+1] - sqr(x[i]))
    return result

def self_weighted_logit(x):
    return 1.0/(1.0 + algopy.exp(-algopy.dot(x,x)))

#############################
# Benchmark utility methods #
#############################
def bench_func(f, x, repeat):
    def wrapped_f():
        return f(x)
    return min(timeit.repeat(wrapped_f, number=1, repeat=repeat))

def bench_fad(f, repeat=5, xlens=(16,160)):
    bench_dict = {'time': [], 'func': [], 'xlen': []}
    g = gradient(f)
    for xlen in xlens:
        x = numpy.random.rand(xlen)
        bench_dict['time'].append(bench_func(f, x, repeat))
        bench_dict['func'].append('f')
        bench_dict['xlen'].append(xlen)
        bench_dict['time'].append(bench_func(g, x, repeat))
        bench_dict['func'].append('g')
        bench_dict['xlen'].append(xlen)
    return bench_dict

script_path = os.path.dirname(os.path.realpath(__file__))

def default_benchmark(*fs):
    folder_path = os.path.join(script_path, 'benchmark_data')
    for f in fs:
        filename = os.path.join(folder_path, f.__name__ + '_times.p')
        with open(filename, 'wb') as file:
            print 'Performing default benchmarks for ' + f.__name__ + '...'
            result = bench_fad(f)
            print '\tdone. Pickling results...'
            pickle.dump(result, file)
            print '\tdone.'
    print 'Done with all benchmarks!'

##################
# Run benchmarks #
##################
def main():
    default_benchmark(ackley, rosenbrock, self_weighted_logit)

if __name__ == '__main__':
    main()