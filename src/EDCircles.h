#ifndef EDCIRCLES_H_
#define EDCIRCLES_H_

#include <cmath>
#include <cstdint>
#include <map>
#include <stack>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CircleFitter.h"
#include "EDPF.h"
#include "LineFitter.h"

#define CLOSE_EDGE_THRES 3.0
#define CIRCLE_FIT_ERR_THRES 1.5

#define MIN_LINE_LEN 10
#define LINE_FIT_ERR_THRES 1.5

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

struct CircleSegment {
  CircleEquation circle;
  std::vector<cv::Point> hops;

  CircleSegment(const CircleEquation& ce, const std::vector<cv::Point>& pts)
      : circle(ce), hops(pts) {}
  CircleSegment() : circle(CircleEquation()), hops(std::vector<cv::Point>()) {}
  ~CircleSegment() {}
};

struct LineSegment {
  LineEquation line;
  std::vector<cv::Point> hops;

  LineSegment(const LineEquation& le, const std::vector<cv::Point>& pts)
      : line(le), hops(pts) {}
  LineSegment() : line(LineEquation()), hops(std::vector<cv::Point>()) {}
  ~LineSegment() {}
};

class EDCircles {
 private:
  std::string src_path_;
  cv::Mat src_img_;
  int32_t height_, width_;

 private:
  std::vector<EdgeSegment> edge_segments_;

  std::vector<CircleSegment> circle_segments_;
  std::map<int32_t, std::vector<LineSegment>> line_segments_;

 private:
  void find_circles_n_lines();

  std::vector<LineSegment> edge_to_lines(const EdgeSegment& edge_segment);
  void edge_to_lines_r(const double* xs,
                       const double* ys,
                       int32_t num_pts,
                       std::vector<LineSegment>& lines);

 public:
  EDCircles(const char* src_path);
  ~EDCircles();

  void show_src_img();
  void show_colored_edges();
  void show_colored_circles();
  void show_colored_lines();
};

#endif