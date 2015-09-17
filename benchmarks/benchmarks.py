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
        result += sqr(b - x[i]) + a*sqr(x[i+1] - sqr(x[i]))
    return result

def self_weighted_logit(x):
    return 1.0/(1.0 + algopy.exp(-algopy.dot(x,x)))

#############################
# Benchmark utility methods #
#############################
folder_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(folder_path, 'benchmark_data')

def data_name(f):
    return f.__name__ + '_times'

def data_filename(f):
    return os.path.join(data_path, data_name(f) + '.p')

def bench_func(f, x, repeat):
    def wrapped_f():
        return f(x)
    return min(timeit.repeat(wrapped_f, number=1, repeat=repeat))

def run_benchmark(f, repeat=5, xlens=(16,1600,16000)):
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

def run_benchmarks(*fs):
    for f in fs:
        with open(data_filename(f), 'wb') as file:
            print 'Performing default benchmarks for ' + f.__name__ + '...'
            result = run_benchmark(f)
            print '\tdone. Pickling results...'
            pickle.dump(result, file)
            print '\tdone.'

def get_benchmark(f):
    with open(data_filename(f), 'rb') as file:
        return pickle.load(file)

def get_benchmarks(*fs):
    return {f.__name__: get_benchmark(f) for f in fs}

##################
# Run benchmarks #
##################
default_fs = (ackley, rosenbrock, self_weighted_logit)

def run_default_benchmarks(): 
    return run_benchmarks(*default_fs)

def get_default_benchmarks(): 
    return get_benchmarks(*default_fs)

if __name__ == '__main__':
    run_default_benchmarks()
