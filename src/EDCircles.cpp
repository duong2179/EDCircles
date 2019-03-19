#include "EDCircles.h"

EDCircles::EDCircles(const char* src_path) : src_path_(src_path) {
  src_img_ = cv::imread(src_path_, 0);

  // EDPF
  EDPF edpf(src_img_);

  // show colored edges
  edpf.show_colored_edges();

  // edge segments
  const std::vector<EdgeChain>& chains = edpf.chains();
  all_edge_segments_ = chains;

  // circle fit
  find_circle_candidates();

  // line fit
  find_line_candidates();

  // make image shown up
  cv::waitKey();
}

EDCircles::~EDCircles() {}

void EDCircles::find_circle_candidates() {
  for (int32_t idx = 0; idx < (int32_t)all_edge_segments_.size(); ++idx) {
    const auto& chain = all_edge_segments_[idx];

    if (!chain.is_closed(CLOSE_EDGE_THRES)) {
      inter_edge_idxes_.push_back(idx);
      continue;
    }

    int32_t edge_len = chain.hops.size();

    std::vector<double> xs, ys;
    xs.reserve(edge_len);
    ys.reserve(edge_len);

    for (const auto& p : chain.hops) {
      xs.push_back(p.y);
      ys.push_back(p.x);
    }

    CircleEquation ce;
    double err = 0.0;
    bool fit = CircleFitter::least_square_fit(xs, ys, ce, err);
    if (fit && err < CIRCLE_FIT_ERR_THRES) {
#ifdef DEBUG_MODE
      std::cout << ce << ", " << edge_len << std::endl;
#endif
      circle_candidates_.emplace_back(ce, chain.hops);
    } else {
      inter_edge_idxes_.push_back(idx);
    }
  }

  auto num_candidates = circle_candidates_.size();
  std::cout << num_candidates << " circle candidates found" << std::endl;
}

void EDCircles::find_line_candidates() {
  for (int32_t idx : inter_edge_idxes_) {
    const auto& chain = all_edge_segments_[idx];

    int32_t edge_len = chain.hops.size();

    std::vector<double> vec_xs, vec_ys;
    vec_xs.reserve(edge_len);
    vec_ys.reserve(edge_len);

    for (const auto& p : chain.hops) {
      vec_xs.push_back(p.y);
      vec_ys.push_back(p.x);
    }

    const double* xs = vec_xs.data();
    const double* ys = vec_ys.data();

    find_line_candidates_r(xs, ys, edge_len);
  }

  auto num_candidates = line_candidates_.size();
  std::cout << num_candidates << " line candidates found" << std::endl;
}

void EDCircles::find_line_candidates_r(const double* xs,
                                       const double* ys,
                                       int32_t num_pts) {
  if (num_pts < MIN_LINE_LEN) {
    return;
  }

  int32_t idx = 0;

  LineEquation line;
  double error = 0.0;
  bool found = false;
  std::vector<cv::Point> hops;

  // find initial line
  while (num_pts >= MIN_LINE_LEN) {
    LineFitter::least_square_fit(xs + idx, ys + idx, MIN_LINE_LEN, line, error);
    if (error <= LINE_FIT_ERR_THRES) {
      for (int32_t i = 0; i < MIN_LINE_LEN; ++i) {
        int32_t row = *(ys + idx + i);
        int32_t col = *(xs + idx + i);
        hops.emplace_back(row, col);
      }
      found = true;
      break;
    } else {
      ++idx;
      --num_pts;
    }
  }

  // couldn't find initial line
  if (!found) {
    return;
  }

  // try adding next point to the line, one by one
  idx += MIN_LINE_LEN;
  int32_t line_len = MIN_LINE_LEN;
  while (line_len < num_pts) {
    double d = LineFitter::distance_to_line(xs[idx], ys[idx], line);
    if (d > LINE_FIT_ERR_THRES) {
      int32_t row = *(ys + idx);
      int32_t col = *(xs + idx);
      hops.emplace_back(row, col);
      break;
    }
    ++idx;
    ++line_len;
  }

#ifdef DEBUG_MODE
  std::cout << line << ", " << line_len << std::endl;
#endif
  line_candidates_.emplace_back(line, hops);

  num_pts -= line_len;
  find_line_candidates_r(xs + idx, ys + idx, num_pts);
}