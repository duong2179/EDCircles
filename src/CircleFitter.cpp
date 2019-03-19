#include "CircleFitter.h"

std::ostream& operator<<(std::ostream& os, const CircleEquation& cir) {
  os << "CircleEquation: (" << cir.xc << "," << cir.yc << "," << cir.rad << ")";
  return os;
}

double CircleFitter::calc_bar(const std::vector<double>& us) {
  double sum = 0.0;
  for (const auto& u : us) {
    sum += u;
  }
  return sum / us.size();
}

std::vector<double> CircleFitter::calc_u(const std::vector<double>& us,
                                         double bar) {
  std::vector<double> diffs(us.size());
  for (const auto& u : us) {
    diffs.push_back(u - bar);
  }
  return diffs;
}

double CircleFitter::calc_suu(const std::vector<double>& us) {
  double sum = 0.0;
  for (const auto& u : us) {
    sum += (u * u);
  }
  return sum;
}

double CircleFitter::calc_suuu(const std::vector<double>& us) {
  double sum = 0.0;
  for (const auto& u : us) {
    sum += (u * u * u);
  }
  return sum;
}

double CircleFitter::calc_suv(const std::vector<double>& us,
                              const std::vector<double>& vs) {
  double sum = 0.0;
  for (int32_t i = 0; i < (int32_t)us.size(); ++i) {
    sum += (us[i] * vs[i]);
  }
  return sum;
}

double CircleFitter::calc_suvv(const std::vector<double>& us,
                               const std::vector<double>& vs) {
  double sum = 0.0;
  for (int32_t i = 0; i < (int32_t)us.size(); ++i) {
    sum += (us[i] * vs[i] * vs[i]);
  }
  return sum;
}

bool CircleFitter::least_square_fit(const std::vector<double>& xs,
                                    const std::vector<double>& ys,
                                    CircleEquation& ce,
                                    double& error) {
  if (xs.size() != ys.size() || xs.size() < 3) {
    return false;
  }

  int32_t N = xs.size();

  double xbar = calc_bar(xs);
  double ybar = calc_bar(ys);

  std::vector<double> us = calc_u(xs, xbar);
  std::vector<double> vs = calc_u(ys, ybar);

  double suu = calc_suu(us);
  double suv = calc_suv(us, vs);
  double svv = calc_suu(vs);

  double suuu = calc_suuu(us);
  double svvv = calc_suuu(vs);

  double suvv = calc_suvv(us, vs);
  double svuu = calc_suvv(vs, us);

  double e4 = 0.5 * (suuu + suvv);
  double e5 = 0.5 * (svvv + svuu);

  double delta = (suu * svv - suv * suv);
  if (std::abs(delta) < 1e-6) {
    return false;
  }
  double uc = (svv * e4 - suv * e5) / delta;
  double vc = (e4 - uc * suu) / suv;

  double xc = uc + xbar;
  double yc = vc + ybar;
  double r = std::sqrt(uc * uc + vc * vc + (suu + svv) / N);
  ce.xc = xc;
  ce.yc = yc;
  ce.rad = r;

  double sum_error = 0.0;
  for (int32_t i = 0; i < (int32_t)xs.size(); ++i) {
    double dx = xs[i] - xc;
    double dy = ys[i] - yc;
    double d = std::sqrt(dx * dx + dy * dy) - r;
    sum_error += d * d;
  }
  error = sqrt(sum_error / N);

  return true;
}