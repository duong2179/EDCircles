#ifndef VECTOR_UTIL_H_
#define VECTOR_UTIL_H_

#include <cmath>

#define FULL_ANGLE 360.0
#define PI 3.141592
#define TWO_PI (2 * PI)

class VectorUtil {
 public:
  double dot_product(double x1, double y1, double x2, double y2);
  double magnitude(double x, double y);
  double angle_bw_two_vectors(double x1, double y1, double x2, double y2);
};

#endif