#include <cstdint>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "EDCircles.h"

int32_t main(int32_t argc, char** argv) {
  if (argc != 2) {
    std::cerr << "[Error] Wrong input." << std::endl;
    exit(1);
  }

  EDCircles edcircles(argv[1]);

  // show colored edges
  edcircles.show_colored_edges();

  // show colored circles
  edcircles.show_colored_circles();

  // show colored lines
  edcircles.show_colored_lines();

  // make image shown up
  cv::waitKey();

  return 0;
}
