#include "LineFitter.h"

double LineFitter::calc_bar(const std::vector<double>& us) {
  double sum = 0.0;
  for (const auto& v : us) {
    sum += v;
  }
  return sum / us.size();
}

bool LineFitter::least_square_fit(const std::vector<double>& xs,
                                  const std::vector<double>& ys,
                                  LineEquation& line,
                                  double& error) {
  if (xs.size() != ys.size() || xs.size() < 2) {
    return false;
  }

  double xbar = calc_bar(xs);
  double ybar = calc_bar(ys);

  double numerator = 0.0;
  double denominator = 0.0;
  for (int32_t i = 0; i < (int32_t)xs.size(); ++i) {
    numerator += (xs[i] - xbar) * (ys[i] - ybar);
    denominator += (xs[i] - xbar) * (xs[i] - xbar);
  }

  double a = numerator / denominator;
  double b = ybar - a * xbar;
  line.a = a;
  line.b = b;

  error = 0.0;

  return true;
}