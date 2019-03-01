#include "EDPF.h"

EDPF::EDPF(const char* src_path) : src_path_(src_path) {
  src_img_ = cv::imread(src_path_, 0);
  height_ = src_img_.rows;
  width_ = src_img_.cols;

  gradient_map_ = new int32_t[height_ * width_]();  // all init-ed to 0
  direction_map_ = new int8_t[height_ * width_]();  // all init-ed to 0
  anchor_map_ = new int8_t[height_ * width_]();     // all init-ed to 0
  edge_map_ = new int8_t[height_ * width_]();       // all init-ed to 0

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
}

int8_t EDPF::smooth_at(int32_t i, int32_t j) {
  return smth_map_[i * width_ + j];
}

int32_t& EDPF::gradient_at(int32_t i, int32_t j) {
  return gradient_map_[i * width_ + j];
}

int8_t& EDPF::direction_at(int32_t i, int32_t j) {
  return direction_map_[i * width_ + j];
}

int8_t& EDPF::anchor_at(int32_t i, int32_t j) {
  return anchor_map_[i * width_ + j];
}

int8_t& EDPF::edge_at(int32_t i, int32_t j) {
  return edge_map_[i * width_ + j];
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
          direction_at(i, j) = DIRECT_VERTL;
        } else {
          direction_at(i, j) = DIRECT_HORZL;
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
      if (direction_at(i, j) == DIRECT_VERTL) {
        int32_t diff_left = gradient_at(i, j) - gradient_at(i, j - 1);
        int32_t diff_right = gradient_at(i, j) - gradient_at(i, j + 1);
        if (diff_left >= ANCHOR_THRES && diff_right >= ANCHOR_THRES) {
          anchor_at(i, j) = 1;
        }
      }
      // horizontal
      else if (direction_at(i, j) == DIRECT_HORZL) {
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
  for (const auto& p : anchors_) {
    int32_t row = p.x;
    int32_t col = p.y;

    // this anchor has been included in some edge
    if (edge_at(row, col) == 1) {
      continue;
    }

    // mark the edge map
    edge_at(row, col) = 1;

    // new edge segment starts
    make_new_chain(p);

    // nodes to traverse for current edge segment
    std::stack<TraverseNode> nodes;

    if (direction_at(row, col) == DIRECT_VERTL) {
      nodes.emplace(row, col, TRAVERSE_UP);
      nodes.emplace(row, col, TRAVERSE_DOWN);
    } else if (direction_at(row, col) == DIRECT_HORZL) {
      nodes.emplace(row, col, TRAVERSE_LEFT);
      nodes.emplace(row, col, TRAVERSE_RIGHT);
    }

    while (!nodes.empty()) {
      TraverseNode node = nodes.top();
      nodes.pop();

      // which direction to traverse
      switch (node.dir) {
        case TRAVERSE_UP:
          traverse_up(nodes, node.row, node.col);
          break;
        case TRAVERSE_DOWN:
          traverse_down(nodes, node.row, node.col);
          break;
        case TRAVERSE_LEFT:
          traverse_left(nodes, node.row, node.col);
          break;
        case TRAVERSE_RIGHT:
          traverse_right(nodes, node.row, node.col);
          break;
      }
    }
  }
}

void EDPF::make_new_chain(const cv::Point& p) {
  int32_t chain_idx = chains_.size();
  cv::Vec3b chain_color = EdgeColors[chain_idx % EdgeColors.size()];
  EdgeChain new_chain(chain_idx, chain_color);
  new_chain.points.push_back(p);
  chains_.push_back(new_chain);
}

void EDPF::append_to_current_chain(ChainEnd end, const cv::Point& p) {
  std::vector<cv::Point>& points = chains_.back().points;
  if (end == ChainEnd::Tail) {
    points.push_back(p);
  } else if (end == ChainEnd::Head) {
    points.insert(points.begin(), p);
  }
}

ChainEnd EDPF::belong_to_current_chain(const cv::Point& p) {
  std::vector<cv::Point>& points = chains_.back().points;
  const cv::Point& lp = points.back();
  const cv::Point& fp = points.front();
  if (std::abs(lp.x - p.x) == 1 || std::abs(lp.y - p.y) == 1) {
    return ChainEnd::Tail;
  } else if (std::abs(fp.x - p.x) == 1 || std::abs(fp.y - p.y) == 1) {
    return ChainEnd::Head;
  }
  return ChainEnd::NA;
}

int32_t EDPF::find_next_hop(const std::vector<cv::Point>& pts) {
  if (pts.empty()) {
    return -1;  // invalid
  }

  int32_t max_i = -1, max_G = 0;
  for (int32_t i = 0; i < (int32_t)pts.size(); ++i) {
    // anchor pixel / traversed edge (closed)
    if (edge_at(pts[i].x, pts[i].y) == 1) {
      return -1;
    }
    // else
    else if (max_G < gradient_at(pts[i].x, pts[i].y)) {
      max_G = gradient_at(pts[i].x, pts[i].y);
      max_i = i;
    }
  }

  return max_i;
}

bool EDPF::move_to_next_hop(const std::vector<cv::Point>& pts,
                            int32_t& row,
                            int32_t& col) {
  int32_t next_idx = find_next_hop(pts);
  if (next_idx == -1) {
    return false;
  }

  const cv::Point& next_pt = pts[next_idx];
  edge_at(next_pt.x, next_pt.y) = 1;

  // next hop is neighboring with tail of current chain
  ChainEnd end = belong_to_current_chain(next_pt);
  if (ChainEnd::NA != end) {
    append_to_current_chain(end, next_pt);
  }
  // new branch
  else {
    make_new_chain(next_pt);
  }

  row = next_pt.x;
  col = next_pt.y;

  return true;
}

bool EDPF::hit_border(int32_t row, int32_t col) {
  return (row <= 1 || row >= height_ - 2 || col <= 1 || col >= width_ - 2);
}

void EDPF::traverse_up(std::stack<TraverseNode>& nodes,
                       int32_t row,
                       int32_t col) {
  // hit border
  if (hit_border(row, col)) {
    return;
  }

  // keep moving up
  while (row > 1 && direction_at(row, col) == DIRECT_VERTL) {
    std::vector<cv::Point> pts = {cv::Point(row - 1, col - 1),
                                  cv::Point(row - 1, col),
                                  cv::Point(row - 1, col + 1)};
    if (!move_to_next_hop(pts, row, col)) {
      return;
    }
  }

  // direction changed
  if (direction_at(row, col) == DIRECT_HORZL) {
    nodes.emplace(row, col, TRAVERSE_LEFT);
    nodes.emplace(row, col, TRAVERSE_RIGHT);
  }
}

void EDPF::traverse_down(std::stack<TraverseNode>& nodes,
                         int32_t row,
                         int32_t col) {
  // hit border
  if (hit_border(row, col)) {
    return;
  }

  // keep moving down
  while (row < height_ - 2 && direction_at(row, col) == DIRECT_VERTL) {
    std::vector<cv::Point> pts = {cv::Point(row + 1, col - 1),
                                  cv::Point(row + 1, col),
                                  cv::Point(row + 1, col + 1)};
    if (!move_to_next_hop(pts, row, col)) {
      return;
    }
  }

  // direction changed
  if (direction_at(row, col) == DIRECT_HORZL) {
    nodes.emplace(row, col, TRAVERSE_LEFT);
    nodes.emplace(row, col, TRAVERSE_RIGHT);
  }
}

void EDPF::traverse_left(std::stack<TraverseNode>& nodes,
                         int32_t row,
                         int32_t col) {
  // hit border
  if (hit_border(row, col)) {
    return;
  }

  // keep moving left
  while (col > 1 && direction_at(row, col) == DIRECT_HORZL) {
    std::vector<cv::Point> pts = {cv::Point(row - 1, col - 1),
                                  cv::Point(row, col - 1),
                                  cv::Point(row + 1, col - 1)};
    if (!move_to_next_hop(pts, row, col)) {
      return;
    }
  }

  // direction changed
  if (direction_at(row, col) == DIRECT_VERTL) {
    nodes.emplace(row, col, TRAVERSE_UP);
    nodes.emplace(row, col, TRAVERSE_DOWN);
  }
}

void EDPF::traverse_right(std::stack<TraverseNode>& nodes,
                          int32_t row,
                          int32_t col) {
  // hit border
  if (hit_border(row, col)) {
    return;
  }

  // keep moving right
  while (col < width_ - 2 && direction_at(row, col) == DIRECT_HORZL) {
    std::vector<cv::Point> pts = {cv::Point(row - 1, col + 1),
                                  cv::Point(row, col + 1),
                                  cv::Point(row + 1, col + 1)};
    if (!move_to_next_hop(pts, row, col)) {
      return;
    }
  }

  // direction changed
  if (direction_at(row, col) == DIRECT_VERTL) {
    nodes.emplace(row, col, TRAVERSE_UP);
    nodes.emplace(row, col, TRAVERSE_DOWN);
  }
}

void EDPF::verify_edges() {}

void EDPF::show_input() {
  cv::imshow("Source Image", src_img_);
}

void EDPF::show_output() {
  cv::Mat output_img = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

  int32_t long_chains = 0;

  for (const auto& chain : chains_) {
    if (chain.points.size() < 10) {
      continue;
    }
    long_chains++;
    for (const auto& p : chain.points) {
      output_img.at<cv::Vec3b>(p.x, p.y) = chain.color;
    }
  }

  std::cout << "No chains: " << chains_.size() << "\n";
  std::cout << "No long chains: " << long_chains << "\n";

  cv::imshow("So far", output_img);
}