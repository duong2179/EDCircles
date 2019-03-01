#include <iostream>

#include "EDPF.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "[Error] Wrong input." << std::endl;
    exit(1);
  }

  EDPF edpf(argv[1]);
  edpf.show_input();
  edpf.show_output();

  cv::waitKey();

  return 0;
}
