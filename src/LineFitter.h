#ifndef LINE_FITTER_H_
#define LINE_FITTER_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// a * x + b * y + c = 0
struct LineEquation {
  double a, b, c;

  LineEquation(double a, double b, double c) : a(a), b(b), c(c) {}
  LineEquation() : LineEquation(0.0, 0.0, 0.0) {}
  ~LineEquation() {}

  friend std::ostream& operator<<(std::ostream& os, const LineEquation& line);
};

// Line fitter
class LineFitter {
 private:
  static double calc_bar(const double* us, int32_t len);
  static double calc_stdev(const double* us, double ubar, int32_t len);

 public:
  static double distance_to_line(double x0,
                                 double y0,
                                 const LineEquation& line);
  static bool least_square_fit(const double* xs,
                               const double* ys,
                               int32_t len,
                               LineEquation& line,
                               double& error);
  static bool least_square_fit(const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               LineEquation& line,
                               double& error);
};

#endif