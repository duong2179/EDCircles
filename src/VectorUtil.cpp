#include "VectorUtil.h"

double VectorUtil::dot_product(double x1, double y1, double x2, double y2) {
  return x1 * x2 + y1 * y2;
}

double VectorUtil::magnitude(double x, double y) {
  return std::sqrt(x * x + y * y);
}

double VectorUtil::angle_bw_two_vectors(double x1,
                                        double y1,
                                        double x2,
                                        double y2) {
  double angle_rad = std::acos(dot_product(x1, y1, x2, y2) /
                               (magnitude(x1, y1) * magnitude(x2, y2)));
  return angle_rad * FULL_ANGLE / TWO_PI;
}