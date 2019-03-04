#ifndef EDCIRCLES_H_
#define EDCIRCLES_H_

#include "CircleFitter.h"
#include "EDPF.h"

#define CLOSE_EDGE_THRES 3.0
#define CIRCLE_FIT_ERR_THRES 1.5

struct CircleCandidate {
  Circle circle;
  std::vector<cv::Point> hops;

  CircleCandidate(const Circle& cir, const std::vector<cv::Point>& pts)
      : circle(cir), hops(pts) {}
  CircleCandidate() : circle(Circle()), hops(std::vector<cv::Point>()) {}
  ~CircleCandidate() {}
};

class EDCircles {
 private:
  std::string src_path_;
  cv::Mat src_img_;

 private:
  std::vector<EdgeChain> edge_segments_;
  std::vector<EdgeChain> remaining_edge_segments_;
  std::vector<CircleCandidate> circle_candidates_;

 private:
  void find_circle_candidates();

 public:
  EDCircles(const char* src_path);
  ~EDCircles();
};

#endif