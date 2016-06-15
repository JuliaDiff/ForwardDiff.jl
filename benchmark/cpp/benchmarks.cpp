#include "benchmarks.h"

double clock_now()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return (double)now.tv_sec + (double)now.tv_usec/1.0e6;
}

template <typename T> T ackley(const std::vector<T> &x) {
    double a = 20.0, b = -0.2, c = 2*M_PI;
    double len_recip = 1.0/x.size();
    T sum_sqrs(0.0);
    T sum_cos(0.0);
    for (auto i : x) {
        sum_cos = sum_cos + cos(c*i);
        sum_sqrs = sum_sqrs + i*i;
    }
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + M_E);
}

template <typename T> T rosenbrock(const std::vector<T> &x) {
    double a = 100.0;
    double b = 1.0;
    T result(0.0);
    for (size_t i = 0; i < x.size()-1; i++) {
        T t1 = b-x[i];
        T t2 = x[i+1]-x[i]*x[i];
        result = result + t1*t1 + a*t2*t2;
    }
    return result;
}

void testgrad(){
    std::vector<double> result12(12);
    std::vector<double> result10(10);

    result12[0] = -2.0;
    result12[1] = -200.0;
    result12[2] = 1002.0;
    result12[3] = 5804.0;
    result12[4] = 16606.0;
    result12[5] = 35808.0;
    result12[6] = 65810.0;
    result12[7] = 109012.0;
    result12[8] = 167814.0;
    result12[9] = 244616.0;
    result12[10] = 341818.0;
    result12[11] = -17800.0;

    result10[0] = -2.0;
    result10[1] = -200.0;
    result10[2] = 1002.0;
    result10[3] = 5804.0;
    result10[4] = 16606.0;
    result10[5] = 35808.0;
    result10[6] = 65810.0;
    result10[7] = 109012.0;
    result10[8] = 167814.0;
    result10[9] = -11000.0;

    std::vector<double> out12(12);
    std::vector<double> out10(10);
    std::vector<double> in12(12);
    std::vector<double> in10(10);

    std::vector<Dual1> dualvec1(10);
    std::vector<Dual2> dualvec2(10);
    std::vector<Dual3> dualvec3(12);
    std::vector<Dual4> dualvec4(12);
    std::vector<Dual5> dualvec5(10);

    for (int i = 0; i < 12; i++) {
        in12[i] = i;
    }

    for (int i = 0; i < 10; i++) {
        in10[i] = i;
    }

    gradient<rosenbrock<Dual1>>(out10, dualvec1, in10);
    assert(result10 == out10);

    gradient<rosenbrock<Dual2>>(out10, dualvec2, in10);
    assert(result10 == out10);

    gradient<rosenbrock<Dual3>>(out12, dualvec3, in12);
    assert(result12 == out12);

    gradient<rosenbrock<Dual4>>(out12, dualvec4, in12);
    assert(result12 == out12);

    gradient<rosenbrock<Dual5>>(out10, dualvec5, in10);
    assert(result10 == out10);
}

int main() {
    testgrad();

    int len = 1*2*3*4*5*100; // must be divisible by all chunk sizes

    std::vector<double> input(len);
    std::vector<double> result1(len);
    std::vector<double> result2(len);
    std::vector<Dual1> dualvec1(len);
    std::vector<Dual2> dualvec2(len);
    std::vector<Dual3> dualvec3(len);
    std::vector<Dual4> dualvec4(len);
    std::vector<Dual5> dualvec5(len);

    for (int i = 0; i < len; i++) {
        input[i] = i + 1;
    }

    // ackley
    std::cout << "benchmarking ackley with vector of length " << len << "..." << std::endl;

    double start = clock_now();
    ackley(input);
    double end = clock_now();
    std::cout << "  took " << end-start << " seconds for value" << std::endl;

    start = clock_now();
    gradient<ackley<Dual1>>(result1, dualvec1, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual1" << std::endl;

    start = clock_now();
    gradient<ackley<Dual2>>(result2, dualvec2, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual2" << std::endl;

    assert(result1 == result2);

    start = clock_now();
    gradient<ackley<Dual3>>(result2, dualvec3, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual3" << std::endl;

    assert(result1 == result2);

    start = clock_now();
    gradient<ackley<Dual4>>(result2, dualvec4, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual4" << std::endl;

    assert(result1 == result2);

    start = clock_now();
    gradient<ackley<Dual5>>(result2, dualvec5, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual5" << std::endl;

    assert(result1 == result2);

    // rosenbrock
    std::cout << "benchmarking rosenbrock with vector of length " << len << "..." << std::endl;

    start = clock_now();
    rosenbrock(input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for value" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual1>>(result1, dualvec1, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual1" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual2>>(result2, dualvec2, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual2" << std::endl;

    assert(result1 == result2);

    start = clock_now();
    gradient<rosenbrock<Dual3>>(result2, dualvec3, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual3" << std::endl;

    assert(result1 == result2);

    start = clock_now();
    gradient<rosenbrock<Dual4>>(result2, dualvec4, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual4" << std::endl;

    assert(result1 == result2);

    start = clock_now();
    gradient<rosenbrock<Dual5>>(result2, dualvec5, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual5" << std::endl;

    assert(result1 == result2);

    return 0;
}
