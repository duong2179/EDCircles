#ifndef LINE_FITTER_H_
#define LINE_FITTER_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// y = a * x + b
struct LineEquation {
  double a, b;

  LineEquation(double a, double b) : a(a), b(b) {}
  LineEquation() : LineEquation(0.0, 0.0) {}
  ~LineEquation() {}
};

// Line fitter
class LineFitter {
 private:
  static double calc_bar(const std::vector<double>& us);

 public:
  static bool least_square_fit(const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               CircleEquation& cir,
                               double& error);
};

#endif