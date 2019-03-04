#include "EDPF.h"

bool EdgeChain::is_closed(double pct_thres) const {
  if (hops.size() < MIN_EDGE_LEN) {
    return false;
  }
  const cv::Point& fp = hops.front();
  const cv::Point& lp = hops.back();
  double del_x = lp.x - fp.x;
  double del_y = lp.y - fp.y;
  double d = std::sqrt(del_x * del_x + del_y * del_y);
  return (d * 100.0 / hops.size() < pct_thres);
}

EDPF::EDPF(const cv::Mat& src_img) : src_img_(src_img) {
  // step 0: init things
  init();

  // step 1: suppress noise
  suppress_noise();

  // step 2: compute gradient and make direction map
  build_gradient_n_direction_map();

  // step 3: scan for anchors
  scan_for_anchors();

  // step 4: link anchors
  draw_edges();

  // step 5: verify by Helmholtz principle
  verify_edges();
}

EDPF::~EDPF() {
  delete[] gradient_map_;
  delete[] direction_map_;
  delete[] anchor_map_;
  delete[] chain_map_;
}

void EDPF::init() {
  height_ = src_img_.rows;
  width_ = src_img_.cols;

  gradient_map_ = new int32_t[height_ * width_]();  // all init-ed to 0
  direction_map_ = new int8_t[height_ * width_]();  // all init-ed to 0
  anchor_map_ = new int8_t[height_ * width_]();     // all init-ed to 0
  chain_map_ = new int32_t[height_ * width_]();     // all init-ed to 0

  current_chain_ = nullptr;
}

int8_t EDPF::smooth_at(int32_t i, int32_t j) {
  return smth_map_[i * width_ + j];
}

int8_t EDPF::smooth_at(const cv::Point& p) {
  return smooth_at(p.x, p.y);
}

int32_t& EDPF::gradient_at(int32_t i, int32_t j) {
  return gradient_map_[i * width_ + j];
}

int32_t& EDPF::gradient_at(const cv::Point& p) {
  return gradient_at(p.x, p.y);
}

int8_t& EDPF::direction_at(int32_t i, int32_t j) {
  return direction_map_[i * width_ + j];
}

int8_t& EDPF::direction_at(const cv::Point& p) {
  return direction_at(p.x, p.y);
}

int8_t& EDPF::anchor_at(int32_t i, int32_t j) {
  return anchor_map_[i * width_ + j];
}

int8_t& EDPF::anchor_at(const cv::Point& p) {
  return anchor_at(p.x, p.y);
}

int32_t& EDPF::chain_at(int32_t i, int32_t j) {
  return chain_map_[i * width_ + j];
}

int32_t& EDPF::chain_at(const cv::Point& p) {
  return chain_at(p.x, p.y);
}

void EDPF::suppress_noise() {
  smth_img_ = cv::Mat(height_, width_, CV_8UC1);
  smth_map_ = smth_img_.data;
  cv::GaussianBlur(src_img_, smth_img_, GAUSS_FILTER, GAUSS_SIGMA);
}

/*
 * [ A B C ]
 * | D x E |
 * [ F G H ]
 *
 * Prewitt:
 *    Gx = (C-A) + (E-D) + (H-F)
 *    Gy = (F-A) + (G-B) + (H-C)
 *
 * Sobel:
 *    Gx = (C-A) + 2*(E-D) + (H-F)
 *    Gy = (F-A) + 2*(G-B) + (H-C)
 */

void EDPF::prewitt_filter(int32_t i, int32_t j, int32_t& Gx, int32_t& Gy) {
  // Gx = (C-A) + (E-D) + (H-F)
  Gx = std::abs((smooth_at(i - 1, j + 1) - smooth_at(i - 1, j - 1)) +
                (smooth_at(i, j + 1) - smooth_at(i, j - 1)) +
                (smooth_at(i + 1, j + 1) - smooth_at(i + 1, j - 1)));
  // Gy = (F-A) + (G-B) + (H-C)
  Gy = std::abs((smooth_at(i + 1, j - 1) - smooth_at(i - 1, j - 1)) +
                (smooth_at(i + 1, j) - smooth_at(i - 1, j)) +
                (smooth_at(i + 1, j + 1) - smooth_at(i - 1, j + 1)));
}

void EDPF::sobel_filter(int32_t i, int32_t j, int32_t& Gx, int32_t& Gy) {
  // Gx = (C-A) + 2 * (E-D) + (H-F)
  Gx = std::abs((smooth_at(i - 1, j + 1) - smooth_at(i - 1, j - 1)) +
                2 * (smooth_at(i, j + 1) - smooth_at(i, j - 1)) +
                (smooth_at(i + 1, j + 1) - smooth_at(i + 1, j - 1)));
  // Gy = (F-A) + 2 * (G-B) + (H-C)
  Gy = std::abs((smooth_at(i + 1, j - 1) - smooth_at(i - 1, j - 1)) +
                2 * (smooth_at(i + 1, j) - smooth_at(i - 1, j)) +
                (smooth_at(i + 1, j + 1) - smooth_at(i - 1, j + 1)));
}

void EDPF::build_gradient_n_direction_map() {
  // set boundaries to all zeros
  // row = 0 & row = height - 1
  for (int32_t j = 0; j < width_; ++j) {
    gradient_at(0, j) = 0.0;
    gradient_at(height_ - 1, j) = 0.0;
  }
  // col = 0 & col = width - 1
  for (int32_t i = 0; i < height_; ++i) {
    gradient_at(i, 0) = 0.0;
    gradient_at(i, width_ - 1) = 0.0;
  }

  // Prewitt filter applied
  for (int32_t i = 1; i < height_ - 1; ++i) {
    for (int32_t j = 1; j < width_ - 1; ++j) {
      int32_t Gx = 0, Gy = 0;
      prewitt_filter(i, j, Gx, Gy);
      int32_t G = (int32_t)std::sqrt((double)(Gx * Gx + Gy * Gy));
      gradient_at(i, j) = G;
      if (gradient_at(i, j) >= GRADIENT_THRES) {
        if (Gx >= Gy) {
          direction_at(i, j) = (int8_t)EdgeDirection::Vertical;
        } else {
          direction_at(i, j) = (int8_t)EdgeDirection::Horizontal;
        }
      }
    }
  }
}

void EDPF::scan_for_anchors() {
  for (int32_t i = 1; i < height_ - 1; ++i) {
    for (int32_t j = 1; j < width_ - 1; ++j) {
      // not on the raster
      if ((i % DETAIL_RATIO != 0) && (j % DETAIL_RATIO != 0)) {
        continue;
      }

      // vertical edge
      if (direction_at(i, j) == (int8_t)EdgeDirection::Vertical) {
        int32_t diff_left = gradient_at(i, j) - gradient_at(i, j - 1);
        int32_t diff_right = gradient_at(i, j) - gradient_at(i, j + 1);
        if (diff_left >= ANCHOR_THRES && diff_right >= ANCHOR_THRES) {
          anchor_at(i, j) = 1;
        }
      }
      // horizontal
      else if (direction_at(i, j) == (int8_t)EdgeDirection::Horizontal) {
        int32_t diff_top = gradient_at(i, j) - gradient_at(i - 1, j);
        int32_t diff_bottom = gradient_at(i, j) - gradient_at(i + 1, j);
        if (diff_top >= ANCHOR_THRES && diff_bottom >= ANCHOR_THRES) {
          anchor_at(i, j) = 1;
        }
      }

      if (anchor_at(i, j) == 1) {
        anchors_.push_back(cv::Point(i, j));
      }
    }
  }
}

// TODO: -> counting sort ???
void EDPF::sort_anchors() {
  std::sort(anchors_.begin(), anchors_.end(),
            [this](const cv::Point& p1, const cv::Point& p2) {
              return this->gradient_at(p1.x, p1.y) >
                     this->gradient_at(p2.x, p2.y);
            });
}

void EDPF::draw_edges() {
  // who knows ?
  if (anchors_.empty()) {
    return;
  }

  // sort anchors in descending order of gradient
  sort_anchors();

  // start with the highest gradient
  for (const auto& anchor : anchors_) {
    // this anchor has been included in some edge
    if (chain_at(anchor) > 0) {
      continue;
    }

    // new edge segment starts
    make_new_chain(anchor);

    // nodes to traverse for current edge segment
    std::stack<DrawNode> nodes;

    if (direction_at(anchor) == (int8_t)EdgeDirection::Vertical) {
      nodes.emplace(anchor, DrawDirection::Up);
      nodes.emplace(anchor, DrawDirection::Down);
    } else if (direction_at(anchor) == (int8_t)EdgeDirection::Horizontal) {
      nodes.emplace(anchor, DrawDirection::Left);
      nodes.emplace(anchor, DrawDirection::Right);
    }

    while (!nodes.empty()) {
      DrawNode node = nodes.top();
      nodes.pop();

      const cv::Point& hop = node.hop;

      bool branched = false;
      ChainEnd end = ChainEnd::NA;

      if (hop == anchor) {
        int32_t anchor_chain_idx = chain_at(anchor);
        current_chain_ = &chains_[anchor_chain_idx - 1];
      }

      // current chain has terminated
      if (current_chain_ == nullptr) {
        branched = true;
        end = ChainEnd::Tail;
      }
      // current chain continued
      else {
        // node is in the middle of current chain -> branch out
        end = which_end_to_grow(hop);
        if (end == ChainEnd::NA) {
          branched = true;
          end = ChainEnd::Tail;
        }
        // node is tail of current chain -> continue growing chain
        else {
          branched = false;
        }
      }

      traverse(nodes, node, branched, end);
    }
  }
}

void EDPF::make_new_chain(const cv::Point& p) {
  int32_t chain_idx = chains_.size() + 1;
  EdgeChain new_chain(chain_idx);
  new_chain.hops.push_back(p);
  chains_.push_back(new_chain);
  current_chain_ = &chains_.back();
  chain_at(p) = chain_idx;
}

void EDPF::grow_current_chain(ChainEnd end, const cv::Point& p) {
  std::vector<cv::Point>& points = current_chain_->hops;
  if (end == ChainEnd::Tail) {
    points.push_back(p);
  } else if (end == ChainEnd::Head) {
    points.insert(points.begin(), p);
  }
}

ChainEnd EDPF::which_end_to_grow(const cv::Point& p) {
  std::vector<cv::Point>& points = current_chain_->hops;
  const cv::Point& lp = points.back();
  const cv::Point& fp = points.front();
  if (lp == p) {
    return ChainEnd::Tail;
  } else if (fp == p) {
    return ChainEnd::Head;
  }
  return ChainEnd::NA;
}

const cv::Point& EDPF::find_best_hop(const std::vector<cv::Point>& pts) {
  int32_t max_i = 0;
  int32_t max_G = gradient_at(pts[max_i]);

  for (int32_t i = 1; i < (int32_t)pts.size(); ++i) {
    if (max_G < gradient_at(pts[i])) {
      max_G = gradient_at(pts[i]);
      max_i = i;
    }
  }

  return pts[max_i];
}

void EDPF::move_to_next_hop(const cv::Point& hop, ChainEnd end) {
  // mark in edge map
  chain_at(hop) = current_chain_->index;

  // append next hop to current chain
  grow_current_chain(end, hop);
}

bool EDPF::hit_border(int32_t row, int32_t col) {
  return (row <= 1 || row >= height_ - 2 || col <= 1 || col >= width_ - 2);
}

bool EDPF::hit_border(const cv::Point& p) {
  return hit_border(p.x, p.y);
}

std::vector<cv::Point> EDPF::neighbors(const cv::Point& point,
                                       DrawDirection dir) {
  int32_t row = point.x;
  int32_t col = point.y;

  std::vector<cv::Point> pts;

  switch (dir) {
    case DrawDirection::Up:
      pts = {cv::Point(row - 1, col - 1), cv::Point(row - 1, col),
             cv::Point(row - 1, col + 1)};
      break;
    case DrawDirection::Down:
      pts = {cv::Point(row + 1, col - 1), cv::Point(row + 1, col),
             cv::Point(row + 1, col + 1)};
      break;
    case DrawDirection::Left:
      pts = {cv::Point(row - 1, col - 1), cv::Point(row, col - 1),
             cv::Point(row + 1, col - 1)};
      break;
    case DrawDirection::Right:
      pts = {cv::Point(row - 1, col + 1), cv::Point(row, col + 1),
             cv::Point(row + 1, col + 1)};
      break;
    default:
      break;
  }

  return pts;
}

EdgeDirection EDPF::draw_tendency(DrawDirection dir) {
  // vertical
  if (dir == DrawDirection::Up || dir == DrawDirection::Down) {
    return EdgeDirection::Vertical;
  }
  // horizontal
  else if (dir == DrawDirection::Left || dir == DrawDirection::Right) {
    return EdgeDirection::Horizontal;
  }
  // unknown
  else {
    return EdgeDirection::NA;
  }
}

bool EDPF::validate_chain_width(const cv::Point& next_hop) {
  int32_t neighbors = 0;
  for (int32_t i = next_hop.x - 1; i <= next_hop.x + 1; ++i) {
    for (int32_t j = next_hop.y - 1; j <= next_hop.y + 1; ++j) {
      if (chain_at(i, j) == current_chain_->index) {
        ++neighbors;
        if (neighbors > 1) {
          return false;
        }
      }
    }
  }
  return true;
}

void EDPF::traverse(std::stack<DrawNode>& nodes,
                    DrawNode node,
                    bool branched,
                    ChainEnd end) {
  DrawDirection dir = node.dir;

  // horizontal / vertical traverse ?
  EdgeDirection tendency = draw_tendency(dir);

  EdgeDirection branch_dir = EdgeDirection::NA;
  DrawDirection turn_dir_1 = DrawDirection::NA;
  DrawDirection turn_dir_2 = DrawDirection::NA;

  // up / down
  if (tendency == EdgeDirection::Vertical) {
    branch_dir = EdgeDirection::Horizontal;
    turn_dir_1 = DrawDirection::Left;
    turn_dir_2 = DrawDirection::Right;
  }
  // left / right
  else if (tendency == EdgeDirection::Horizontal) {
    branch_dir = EdgeDirection::Vertical;
    turn_dir_1 = DrawDirection::Up;
    turn_dir_2 = DrawDirection::Down;
  }
  // weird
  else {
    return;
  }

  cv::Point hop = node.hop;

  // keep moving up
  while (!hit_border(hop) && direction_at(hop) == (int8_t)tendency) {
    std::vector<cv::Point> pts = neighbors(hop, dir);
    const cv::Point& next_hop = find_best_hop(pts);
    // hit an existing edge ?
    if (chain_at(next_hop) > 0) {
      return;
    }
    // move to next hop
    else {
      if (branched) {
        make_new_chain(next_hop);
        branched = false;
      } else {
        if (!validate_chain_width(next_hop)) {
          return;
        } else {
          move_to_next_hop(next_hop, end);
        }
      }

      hop = next_hop;
    }
  }

  // hit border
  if (hit_border(hop)) {
    current_chain_ = nullptr;
  }
  // direction changed
  else if (direction_at(hop) == (int8_t)branch_dir) {
    nodes.emplace(hop, turn_dir_1);
    nodes.emplace(hop, turn_dir_2);
  }
  // no direction
  else {
    current_chain_ = nullptr;
  }
}

void EDPF::verify_edges() {
  // determine short chains
  std::vector<int32_t> short_chains;
  for (int32_t i = 0; i < (int32_t)chains_.size(); ++i) {
    const EdgeChain& chain = chains_[i];
    if (chain.hops.size() < MIN_EDGE_LEN) {
      short_chains.push_back(i);
      for (const auto& p : chain.hops) {
        chain_at(p) = 0;
      }
    }
  }
  // remove short chains
  for (int32_t i = short_chains.size() - 1; i >= 0; --i) {
    int32_t j = short_chains[i];
    chains_.erase(chains_.begin() + j);
  }
  std::cout << "Removed " << short_chains.size() << " short chains"
            << std::endl;

  std::cout << "Skipping Helmholtz principle validation..." << std::endl;
}

const std::vector<EdgeChain>& EDPF::chains() {
  return chains_;
}

void EDPF::show_colored_edges() {
  std::vector<cv::Vec3b> colors = {cv::Vec3b(255, 255, 255),   // White
                                   cv::Vec3b(255, 0, 0),       // Blue
                                   cv::Vec3b(0, 255, 0),       // Lime
                                   cv::Vec3b(0, 0, 255),       // Red
                                   cv::Vec3b(0, 255, 255),     // Yellow
                                   cv::Vec3b(255, 255, 0),     // Cyan
                                   cv::Vec3b(255, 0, 255),     // Magenta
                                   cv::Vec3b(0, 0, 128),       // Maroon
                                   cv::Vec3b(0, 128, 128),     // Olive
                                   cv::Vec3b(128, 128, 128)};  // Gray

  cv::Mat output_img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

  int32_t counter = 0;
  for (const auto& chain : chains_) {
    cv::Vec3b color = colors[counter++ % colors.size()];
    for (const auto& p : chain.hops) {
      output_img.at<cv::Vec3b>(p.x, p.y) = color;
    }
  }

  std::cout << "No. chains: " << chains_.size() << std::endl;
  cv::imshow("So far", output_img);
}