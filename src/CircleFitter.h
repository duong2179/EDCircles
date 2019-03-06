#ifndef CIRCLE_FITTER_H_
#define CIRCLE_FITTER_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// x^2 + y^2 = rad^2
struct CircleEquation {
  double xc, yc, rad;

  CircleEquation(double x, double y, double r) : xc(x), yc(y), rad(r) {}
  CircleEquation() : CircleEquation(0.0, 0.0, 0.0) {}
  ~CircleEquation() {}

  friend std::ostream& operator<<(std::ostream& os, const CircleEquation& cir);
};

// Circle fitter (refer to [64])
class CircleFitter {
 private:
  static double calc_bar(const std::vector<double>& us);
  static std::vector<double> calc_u(const std::vector<double>& us, double bar);
  static double calc_suu(const std::vector<double>& us);
  static double calc_suuu(const std::vector<double>& us);
  static double calc_suv(const std::vector<double>& us,
                         const std::vector<double>& vs);
  static double calc_suvv(const std::vector<double>& us,
                          const std::vector<double>& vs);

 public:
  static bool least_square_fit(const std::vector<double>& xs,
                               const std::vector<double>& ys,
                               CircleEquation& cir,
                               double& error);
};

#endif