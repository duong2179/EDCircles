#ifndef EDCIRCLES_H_
#define EDCIRCLES_H_

#include "CircleFitter.h"
#include "EDPF.h"
#include "LineFitter.h"

#define CLOSE_EDGE_THRES 3.0
#define CIRCLE_FIT_ERR_THRES 1.5

#define MIN_LINE_LEN 10
#define LINE_FIT_ERR_THRES 2.0

struct CircleCandidate {
  CircleEquation circle;
  std::vector<cv::Point> hops;

  CircleCandidate(const CircleEquation& ce, const std::vector<cv::Point>& pts)
      : circle(ce), hops(pts) {}
  CircleCandidate()
      : circle(CircleEquation()), hops(std::vector<cv::Point>()) {}
  ~CircleCandidate() {}
};

struct LineCandidate {
  LineEquation line;
  std::vector<cv::Point> hops;

  LineCandidate(const LineEquation& le, const std::vector<cv::Point>& pts)
      : line(le), hops(pts) {}
  LineCandidate() : line(LineEquation()), hops(std::vector<cv::Point>()) {}
  ~LineCandidate() {}
};

class EDCircles {
 private:
  std::string src_path_;
  cv::Mat src_img_;

 private:
  std::vector<EdgeChain> all_edge_segments_;

  std::vector<int32_t> inter_edge_idxes_;

  std::vector<CircleCandidate> circle_candidates_;
  std::vector<LineCandidate> line_candidates_;

 private:
  void find_circle_candidates();
  void find_line_candidates();
  void find_line_candidates_r(const double* xs,
                              const double* ys,
                              int32_t num_pts);

 public:
  EDCircles(const char* src_path);
  ~EDCircles();
};

#endif