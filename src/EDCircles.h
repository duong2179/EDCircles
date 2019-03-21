#ifndef EDCIRCLES_H_
#define EDCIRCLES_H_

#include <cmath>
#include <cstdint>
#include <stack>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CircleFitter.h"
#include "EDPF.h"
#include "LineFitter.h"

#define CLOSE_EDGE_THRES 3.0
#define CIRCLE_FIT_ERR_THRES 1.5

#define MIN_LINE_LEN 10
#define LINE_FIT_ERR_THRES 2.0

static std::vector<cv::Vec3b> XColors = {cv::Vec3b(255, 255, 255),   // White
                                         cv::Vec3b(255, 0, 0),       // Blue
                                         cv::Vec3b(0, 255, 0),       // Lime
                                         cv::Vec3b(0, 0, 255),       // Red
                                         cv::Vec3b(0, 255, 255),     // Yellow
                                         cv::Vec3b(255, 255, 0),     // Cyan
                                         cv::Vec3b(255, 0, 255),     // Magenta
                                         cv::Vec3b(0, 0, 128),       // Maroon
                                         cv::Vec3b(0, 128, 128),     // Olive
                                         cv::Vec3b(128, 128, 128)};  // Gray

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
  int32_t height_, width_;

 private:
  std::vector<EdgeSegment> all_edge_segments_;
  std::vector<int32_t> intmd_edge_idxes_;

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

  void show_src_img();
  void show_colored_edges();
  void show_colored_circles();
  void show_colored_lines();
};

#endif