#include "EDCircles.h"

EDCircles::EDCircles(const char* src_path) : src_path_(src_path) {
  src_img_ = cv::imread(src_path_, 0);

  // height & width
  height_ = src_img_.rows;
  width_ = src_img_.cols;

  // EDPF
  EDPF edpf(src_img_);

  // edge segments
  edge_segments_ = edpf.chains();

  // edge segments -> circles & lines
  find_circles_n_lines();
}

EDCircles::~EDCircles() {}

void EDCircles::find_circles_n_lines() {
  for (const auto& edge_segment : edge_segments_) {
    int32_t edge_idx = edge_segment.index;
    int32_t edge_len = edge_segment.hops.size();

    if (!edge_segment.is_closed(CLOSE_EDGE_THRES)) {
      line_segments_[edge_idx] = edge_to_lines(edge_segment);
      continue;
    }

    std::vector<double> xs, ys;
    xs.reserve(edge_len);
    ys.reserve(edge_len);

    for (const auto& p : edge_segment.hops) {
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
      circle_segments_.emplace_back(ce, edge_segment.hops);
    } else {
      line_segments_[edge_idx] = edge_to_lines(edge_segment);
    }
  }

  int32_t num_circles = circle_segments_.size();
  std::cout << num_circles << " circle candidates found" << std::endl;

  int32_t num_lines = 0;
  for (const auto& kv : line_segments_) {
    const auto& lines_per_edge = kv.second;
    num_lines += lines_per_edge.size();
  }
  std::cout << num_lines << " line candidates found" << std::endl;
}

std::vector<LineSegment> EDCircles::edge_to_lines(
    const EdgeSegment& edge_segment) {
  int32_t edge_len = edge_segment.hops.size();

  std::vector<double> vec_xs, vec_ys;
  vec_xs.reserve(edge_len);
  vec_ys.reserve(edge_len);

  for (const auto& p : edge_segment.hops) {
    vec_xs.push_back(p.y);
    vec_ys.push_back(p.x);
  }

  const double* xs = vec_xs.data();
  const double* ys = vec_ys.data();

  std::vector<LineSegment> lines;
  edge_to_lines_r(xs, ys, edge_len, lines);

  return lines;
}

void EDCircles::edge_to_lines_r(const double* xs,
                                const double* ys,
                                int32_t num_pts,
                                std::vector<LineSegment>& lines) {
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
      break;
    }

    int32_t row = *(ys + idx);
    int32_t col = *(xs + idx);
    hops.emplace_back(row, col);

    ++idx;
    ++line_len;
  }

#ifdef DEBUG_MODE
  std::cout << line << ", " << line_len << std::endl;
#endif
  lines.emplace_back(line, hops);

  num_pts -= line_len;
  edge_to_lines_r(xs + idx, ys + idx, num_pts, lines);
}

void EDCircles::show_src_img() {
  cv::imshow("Source Image", src_img_);
}

void EDCircles::show_colored_edges() {
  cv::Mat output_img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

  int32_t counter = 0;
  for (const auto& chain : edge_segments_) {
    cv::Vec3b color = XColors[counter++ % XColors.size()];
    for (const auto& p : chain.hops) {
      output_img.at<cv::Vec3b>(p.x, p.y) = color;
    }
  }

  cv::imshow("Colored Edges", output_img);
}

void EDCircles::show_colored_circles() {
  cv::Mat output_img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

  int32_t counter = 0;
  for (const auto& circle : circle_segments_) {
    cv::Vec3b color = XColors[counter++ % XColors.size()];
    for (const auto& p : circle.hops) {
      output_img.at<cv::Vec3b>(p.x, p.y) = color;
    }
  }

  cv::imshow("Colored Circles", output_img);
}

void EDCircles::show_colored_lines() {
  cv::Mat output_img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

  int32_t counter = 0;
  for (const auto& kv : line_segments_) {
    const auto& lines_per_edge = kv.second;
    for (const auto& line : lines_per_edge) {
      cv::Vec3b color = XColors[counter++ % XColors.size()];
      for (const auto& p : line.hops) {
        output_img.at<cv::Vec3b>(p.x, p.y) = color;
      }
    }
  }

  cv::imshow("Colored Lines", output_img);
}