#ifndef __INCLUDED__
#define __INCLUDED__

#include <iostream>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include <assert.h>

class Dual1 {
public:
    double real;
    double eps1;
    Dual1(double real, double eps1);
    Dual1(double real);
    Dual1();
    friend std::ostream& operator<<(std::ostream& os, const Dual1& dt);
};

Dual1 operator +(const Dual1& x, const Dual1& y);
Dual1 operator -(const Dual1& x, const Dual1& y);
Dual1 operator *(const Dual1& x, const Dual1& y);
Dual1 operator *(double x, const Dual1 &y);
Dual1 operator -(double x, const Dual1 &y);
Dual1 sin(const Dual1& x);
Dual1 cos(const Dual1& x);
Dual1 sqrt(const Dual1& x);
Dual1 exp(const Dual1& x);

class Dual2 {
public:
    double real;
    double eps1;
    double eps2;
    Dual2(double real, double eps1, double eps2);
    Dual2(double real);
    Dual2();
    friend std::ostream& operator<<(std::ostream& os, const Dual2& dt);
};

Dual2 operator +(const Dual2& x, const Dual2& y);
Dual2 operator -(const Dual2& x, const Dual2& y);
Dual2 operator *(const Dual2& x, const Dual2& y);
Dual2 operator *(double x, const Dual2 &y);
Dual2 operator -(double x, const Dual2 &y);
Dual2 sin(const Dual2& x);
Dual2 cos(const Dual2& x);
Dual2 sqrt(const Dual2& x);
Dual2 exp(const Dual2& x);

class Dual3 {
public:
    double real;
    double eps1;
    double eps2;
    double eps3;
    Dual3(double real, double eps1, double eps2, double eps3);
    Dual3(double real);
    Dual3();
    friend std::ostream& operator<<(std::ostream& os, const Dual3& dt);
};

Dual3 operator +(const Dual3& x, const Dual3& y);
Dual3 operator -(const Dual3& x, const Dual3& y);
Dual3 operator *(const Dual3& x, const Dual3& y);
Dual3 operator *(double x, const Dual3 &y);
Dual3 operator -(double x, const Dual3 &y);
Dual3 sin(const Dual3& x);
Dual3 cos(const Dual3& x);
Dual3 sqrt(const Dual3& x);
Dual3 exp(const Dual3& x);

class Dual4 {
public:
    double real;
    double eps1;
    double eps2;
    double eps3;
    double eps4;
    Dual4(double real, double eps1, double eps2, double eps3, double eps4);
    Dual4(double real);
    Dual4();
    friend std::ostream& operator<<(std::ostream& os, const Dual4& dt);
};

Dual4 operator +(const Dual4& x, const Dual4& y);
Dual4 operator -(const Dual4& x, const Dual4& y);
Dual4 operator *(const Dual4& x, const Dual4& y);
Dual4 operator *(double x, const Dual4 &y);
Dual4 operator -(double x, const Dual4 &y);
Dual4 sin(const Dual4& x);
Dual4 cos(const Dual4& x);
Dual4 sqrt(const Dual4& x);
Dual4 exp(const Dual4& x);

class Dual5 {
public:
    double real;
    double eps1;
    double eps2;
    double eps3;
    double eps4;
    double eps5;
    Dual5(double real, double eps1, double eps2, double eps3, double eps4, double eps5);
    Dual5(double real);
    Dual5();
    friend std::ostream& operator<<(std::ostream& os, const Dual5& dt);
};

Dual5 operator +(const Dual5& x, const Dual5& y);
Dual5 operator -(const Dual5& x, const Dual5& y);
Dual5 operator *(const Dual5& x, const Dual5& y);
Dual5 operator *(double x, const Dual5 &y);
Dual5 operator -(double x, const Dual5 &y);
Dual5 sin(const Dual5& x);
Dual5 cos(const Dual5& x);
Dual5 sqrt(const Dual5& x);
Dual5 exp(const Dual5& x);

template <Dual1 (*F)(const std::vector<Dual1> &)> void gradient(std::vector<double> &result, std::vector<Dual1> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual1(input[i]);
    }
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual1(input[i], 1.0);
        Dual1 val = F(dualvec);
        result[i] = val.eps1;
        dualvec[i] = Dual1(input[i]);
    }
}

template <Dual2 (*F)(const std::vector<Dual2> &)> void gradient(std::vector<double> &result, std::vector<Dual2> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual2(input[i]);
    }
    for (int i = 0; i < (len / 2); i++) {
        int j1 = i*2;
        int j2 = j1+1;
        dualvec[j1] = Dual2(input[j1], 1.0, 0.0);
        dualvec[j2] = Dual2(input[j2], 0.0, 1.0);
        Dual2 val = F(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        dualvec[j1] = Dual2(input[j1]);
        dualvec[j2] = Dual2(input[j2]);
    }
}

template <Dual3 (*F)(const std::vector<Dual3> &)> void gradient(std::vector<double> &result, std::vector<Dual3> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual3(input[i]);
    }
    for (int i = 0; i < (len / 3); i++) {
        int j1 = i*3;
        int j2 = j1+1;
        int j3 = j1+2;
        dualvec[j1] = Dual3(input[j1], 1.0, 0.0, 0.0);
        dualvec[j2] = Dual3(input[j2], 0.0, 1.0, 0.0);
        dualvec[j3] = Dual3(input[j3], 0.0, 0.0, 1.0);
        Dual3 val = F(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        result[j3] = val.eps3;
        dualvec[j1] = Dual3(input[j1]);
        dualvec[j2] = Dual3(input[j2]);
        dualvec[j3] = Dual3(input[j3]);
    }
}

template <Dual4 (*F)(const std::vector<Dual4> &)> void gradient(std::vector<double> &result, std::vector<Dual4> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual4(input[i]);
    }
    for (int i = 0; i < (len / 4); i++) {
        int j1 = i*4;
        int j2 = j1+1;
        int j3 = j1+2;
        int j4 = j1+3;
        dualvec[j1] = Dual4(input[j1], 1.0, 0.0, 0.0, 0.0);
        dualvec[j2] = Dual4(input[j2], 0.0, 1.0, 0.0, 0.0);
        dualvec[j3] = Dual4(input[j3], 0.0, 0.0, 1.0, 0.0);
        dualvec[j4] = Dual4(input[j4], 0.0, 0.0, 0.0, 1.0);
        Dual4 val = F(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        result[j3] = val.eps3;
        result[j4] = val.eps4;
        dualvec[j1] = Dual4(input[j1]);
        dualvec[j2] = Dual4(input[j2]);
        dualvec[j3] = Dual4(input[j3]);
        dualvec[j4] = Dual4(input[j4]);
    }
}

template <Dual5 (*F)(const std::vector<Dual5> &)> void gradient(std::vector<double> &result, std::vector<Dual5> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual5(input[i]);
    }
    for (int i = 0; i < (len / 5); i++) {
        int j1 = i*5;
        int j2 = j1+1;
        int j3 = j1+2;
        int j4 = j1+3;
        int j5 = j1+4;
        dualvec[j1] = Dual5(input[j1], 1.0, 0.0, 0.0, 0.0, 0.0);
        dualvec[j2] = Dual5(input[j2], 0.0, 1.0, 0.0, 0.0, 0.0);
        dualvec[j3] = Dual5(input[j3], 0.0, 0.0, 1.0, 0.0, 0.0);
        dualvec[j4] = Dual5(input[j4], 0.0, 0.0, 0.0, 1.0, 0.0);
        dualvec[j5] = Dual5(input[j5], 0.0, 0.0, 0.0, 0.0, 1.0);
        Dual5 val = F(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        result[j3] = val.eps3;
        result[j4] = val.eps4;
        result[j5] = val.eps5;
        dualvec[j1] = Dual5(input[j1]);
        dualvec[j2] = Dual5(input[j2]);
        dualvec[j3] = Dual5(input[j3]);
        dualvec[j4] = Dual5(input[j4]);
        dualvec[j5] = Dual5(input[j5]);
    }
}

#endif
