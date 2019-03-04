#include <iostream>

#include "EDCircles.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "[Error] Wrong input." << std::endl;
    exit(1);
  }

  EDCircles edcircles(argv[1]);

  return 0;
}
