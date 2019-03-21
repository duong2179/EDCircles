#ifndef EDPF_H_
#define EDPF_H_

// Parameter free for EDPF (refer to [52])
#define GAUSS_FILTER cv::Size(5, 5)
#define GAUSS_SIGMA 1.0
#define GRADIENT_THRES 25
#define ANCHOR_THRES 3
#define DETAIL_RATIO 1

#define MIN_EDGE_LEN 10

#include <cmath>
#include <cstdint>
#include <stack>
#include <vector>

#include <opencv2/opencv.hpp>

// Edge direction
enum class EdgeDirection { NA = 0, Vertical, Horizontal };

// Which direction to traverse while drawing edges
enum class DrawDirection { NA = 0, Up, Down, Left, Right };

// To which end the next hop will be added to current chain
enum class ChainEnd { NA = 0, Head, Tail };

struct DrawNode {
  cv::Point hop;
  DrawDirection dir;

  DrawNode(int32_t x, int32_t y, DrawDirection d) : hop(x, y), dir(d) {}
  DrawNode(const cv::Point& p, DrawDirection d) : hop(p), dir(d) {}
};

struct EdgeSegment {
  int32_t index;
  std::vector<cv::Point> hops;

  EdgeSegment(int32_t i) : index(i) {}
  bool is_closed(double pct_thres) const;
};

class EDPF {
 private:
  cv::Mat src_img_;
  cv::Mat smth_img_;
  int32_t height_, width_;
  uint8_t* smth_map_;
  int32_t* gradient_map_;
  int8_t* direction_map_;
  int8_t* anchor_map_;
  int32_t* chain_map_;

  std::vector<cv::Point> anchors_;
  std::vector<EdgeSegment> chains_;
  EdgeSegment* current_chain_;

 private:
  EDPF(const EDPF&) = delete;
  EDPF& operator=(const EDPF&) = delete;

 private:
  void init();
  void clean();

  int8_t smooth_at(int32_t i, int32_t j);
  int8_t smooth_at(const cv::Point& p);

  int32_t& gradient_at(int32_t i, int32_t j);
  int32_t& gradient_at(const cv::Point& p);

  int8_t& direction_at(int32_t i, int32_t j);
  int8_t& direction_at(const cv::Point& p);

  int8_t& anchor_at(int32_t i, int32_t j);
  int8_t& anchor_at(const cv::Point& p);

  int32_t& chain_at(int32_t i, int32_t j);
  int32_t& chain_at(const cv::Point& p);

  void prewitt_filter(int32_t i, int32_t j, int32_t& Gx, int32_t& Gy);
  void sobel_filter(int32_t i, int32_t j, int32_t& Gx, int32_t& Gy);

  void sort_anchors();

  void traverse(std::stack<DrawNode>& nodes,
                DrawNode node,
                bool branched,
                ChainEnd end);
  EdgeDirection draw_tendency(DrawDirection dir);

  bool hit_border(int32_t row, int32_t col);
  bool hit_border(const cv::Point& p);

  std::vector<cv::Point> neighbors(const cv::Point& hop, DrawDirection dir);
  std::vector<cv::Point> neighbors(const cv::Point& hop);
  const cv::Point& find_best_hop(const std::vector<cv::Point>& pts);
  bool validate_chain_width(const cv::Point& next_hop);
  void move_to_next_hop(const cv::Point& hop, ChainEnd end);
  void make_new_chain(const cv::Point& hop);
  void grow_current_chain(ChainEnd end, const cv::Point& hop);
  ChainEnd which_end_to_grow(const cv::Point& p);

  void eliminate_short_chains();
  void reindex_chains();

  void suppress_noise();
  void build_gradient_n_direction_map();
  void scan_for_anchors();
  void draw_edges();
  void verify_edges();

 public:
  EDPF(const cv::Mat& src_img);
  ~EDPF();

  const std::vector<EdgeSegment>& chains();
};

#endif
