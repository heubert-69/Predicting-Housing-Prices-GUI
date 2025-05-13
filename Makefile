CXX = g++
CXXFLAGS = -std=c++17 -O2

# Qt6 flags via pkg-config
QT_CFLAGS  = $(shell pkg-config --cflags Qt6Widgets)
QT_LIBS    = $(shell pkg-config --libs Qt6Widgets)

# ONNX Runtime paths
ONNX_INC = ./lib/onnxruntime/include
ONNX_LIB = ./lib/onnxruntime/lib

INCLUDES    = -I$(ONNX_INC)
LIBS        = -L$(ONNX_LIB) -lonnxruntime

# Embed runtime path so libonnxruntime.so can be found relative to the binary
RPATH_FLAGS = -Wl,-rpath,'$$ORIGIN/$(ONNX_LIB)'

SRC    = main.cpp infer.cpp
OBJ    = $(SRC:.cpp=.o)
TARGET = predictor

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ \
		$(INCLUDES) $(LIBS) $(QT_CFLAGS) $(QT_LIBS) $(RPATH_FLAGS)

%.o: %.cpp infer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES) $(QT_CFLAGS)

clean:
	rm -f $(OBJ) $(TARGET)
