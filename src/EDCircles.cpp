#include "EDCircles.h"

EDCircles::EDCircles(const char* src_path) : src_path_(src_path) {
  src_img_ = cv::imread(src_path_, 0);

  // EDPF
  EDPF edpf(src_img_);

  // show colored edges
  edpf.show_colored_edges();

  // edge segments
  const std::vector<EdgeChain>& chains = edpf.chains();
  edge_segments_ = chains;

  // circle fit
  find_circle_candidates();

  // make image shown up
  cv::waitKey();
}

EDCircles::~EDCircles() {}

void EDCircles::find_circle_candidates() {
  for (const auto& chain : edge_segments_) {
    if (!chain.is_closed(CLOSE_EDGE_THRES)) {
      remaining_edge_segments_.push_back(chain);
      continue;
    }

    std::vector<double> xs, ys;
    for (const auto& p : chain.hops) {
      xs.push_back(p.y);
      ys.push_back(p.x);
    }

    Circle cir;
    double err = 0.0;
    bool fit = CircleFitter::least_square_fit(xs, ys, cir, err);
    if (fit && err < CIRCLE_FIT_ERR_THRES) {
      std::cout << "Circle candidate: " << cir << ", err: " << err << std::endl;
      circle_candidates_.emplace_back(cir, chain.hops);
    } else {
      remaining_edge_segments_.push_back(chain);
    }
  }

  std::cout << circle_candidates_.size() << " circle candidates found"
            << std::endl;
}
