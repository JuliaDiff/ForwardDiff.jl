#ifndef __INCLUDED__
#define __INCLUDED__

#include <iostream>
#include <cmath>
#include <vector>
#include <sys/time.h>

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

void gradient(std::function<Dual1(std::vector<Dual1>)> f, std::vector<double> &result, std::vector<Dual1> &dualvec, const std::vector<double> &input);
void gradient(std::function<Dual2(std::vector<Dual2>)> f, std::vector<double> &result, std::vector<Dual2> &dualvec, const std::vector<double> &input);
void gradient(std::function<Dual3(std::vector<Dual3>)> f, std::vector<double> &result, std::vector<Dual3> &dualvec, const std::vector<double> &input);
void gradient(std::function<Dual4(std::vector<Dual4>)> f, std::vector<double> &result, std::vector<Dual4> &dualvec, const std::vector<double> &input);
void gradient(std::function<Dual5(std::vector<Dual5>)> f, std::vector<double> &result, std::vector<Dual5> &dualvec, const std::vector<double> &input);

#endif
