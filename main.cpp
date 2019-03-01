#include <iostream>
#include <opencv2/opencv.hpp>

#include "EDPF.h"

using namespace std;

int main(int argc, char** argv) {
  cv::Mat src_img = cv::imread(argv[1], 0);
  cv::imshow("Source Image", src_img);

  EDPF edpf(src_img);
  edpf.show_output();

  cv::waitKey();

  return 0;
}
