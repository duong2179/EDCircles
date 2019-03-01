CXX := g++
CXX_FLAGS := -std=c++11 -O3 -Wall -Wextra -Werror -Wno-format-zero-length -pedantic -Isrc

CV_INCS = -I../opencv-4.0.0/include/opencv4
CV_LIBS = -L../opencv-4.0.0/lib -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_imgproc

APP_DIR := .
COMM_DIR := src
OBJ_DIR := obj

APP_FILES := $(wildcard $(APP_DIR)/*.cpp)
COMM_FILES := $(wildcard $(COMM_DIR)/*.cpp)

APP_OBJS := $(patsubst $(APP_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(APP_FILES))
COMM_OBJS := $(patsubst $(COMM_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMM_FILES))

all: mkdir edcircles

mkdir:
	mkdir -p $(OBJ_DIR) 

edcircles: $(APP_OBJS) $(COMM_OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CV_LIBS)

$(OBJ_DIR)/%.o: $(APP_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(CV_INCS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(COMM_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(CV_INCS) -c -o $@ $<

clean:
	rm $(OBJ_DIR)/*.o
	rmdir $(OBJ_DIR)
	rm edcircles

.PHONY: clean
