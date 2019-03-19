#include "LineFitter.h"

std::ostream& operator<<(std::ostream& os, const LineEquation& line) {
  os << "LineFitter: (" << line.a << "," << line.b << "," << line.c << ")";
  return os;
}

double LineFitter::calc_bar(const double* us, int32_t len) {
  double sum = 0.0;
  for (int32_t i = 0; i < len; ++i) {
    sum += us[i];
  }
  return sum / len;
}

double LineFitter::calc_stdev(const double* us, double ubar, int32_t len) {
  double squared_sum = 0.0;
  for (int32_t i = 0; i < len; ++i) {
    squared_sum += (us[i] - ubar) * (us[i] - ubar);
  }
  return std::sqrt(squared_sum / (len - 1));
}

bool LineFitter::least_square_fit(const double* xs,
                                  const double* ys,
                                  int32_t len,
                                  LineEquation& line,
                                  double& error) {
  if (len < 2) {
    return false;
  }

  double xbar = calc_bar(xs, len);
  double ybar = calc_bar(ys, len);

  double stdev_x = calc_stdev(xs, xbar, len);
  double stdev_y = calc_stdev(ys, ybar, len);

  bool inverted = false;

  if (stdev_y > stdev_x) {
    inverted = true;
    // swap xs ys
    const double* ts = xs;
    xs = ys;
    ys = ts;
    // swap xbar ybar
    double tbar = xbar;
    xbar = ybar;
    ybar = tbar;
  }

  double numerator = 0.0;
  double denominator = 0.0;
  for (int32_t i = 0; i < len; ++i) {
    numerator += (xs[i] - xbar) * (ys[i] - ybar);
    denominator += (xs[i] - xbar) * (xs[i] - xbar);
  }

  double a = numerator / denominator;
  double b = ybar - a * xbar;

  // y = a * x + b -> a * x + (-1) * y + b = 0
  if (!inverted) {
    line.a = a;
    line.b = -1.0;
    line.c = b;
  }
  // x = a * y + b -> (-1) * x + a * y + b = 0
  else {
    line.a = -1.0;
    line.b = a;
    line.c = b;
  }

  // if swapped -> revert
  if (inverted) {
    // swap xs ys
    const double* ts = xs;
    xs = ys;
    ys = ts;
    // swap xbar ybar
    double tbar = xbar;
    xbar = ybar;
    ybar = tbar;
  }

  double ss_error = 0.0;
  for (int32_t i = 0; i < len; ++i) {
    double d = distance_to_line(xs[i], ys[i], line);
    ss_error += d * d;
  }
  error = ss_error / len;

  return true;
}

bool LineFitter::least_square_fit(const std::vector<double>& xs,
                                  const std::vector<double>& ys,
                                  LineEquation& line,
                                  double& error) {
  const double* xdata = xs.data();
  const double* ydata = ys.data();
  int32_t len = xs.size();
  return least_square_fit(xdata, ydata, len, line, error);
}

double LineFitter::distance_to_line(double x0,
                                    double y0,
                                    const LineEquation& line) {
  double numerator = std::abs(line.a * x0 + line.b * y0 + line.c);
  double denominator = std::sqrt(line.a * line.a + line.b * line.b);
  return numerator / denominator;
}