# MultiSim Makefile
# C++17 with ODE physics engine for wheeled quadruped simulation.

# === Compiler Settings ===
CXX := clang++
CXXFLAGS := -std=c++17 -stdlib=libc++ -Wall -Wextra -Wpedantic -O2
CXXFLAGS += -Wno-deprecated-declarations  # Silence OpenGL deprecation warnings on macOS

# === Directories ===
INCLUDE_DIR := include
SRC_DIR := src
BUILD_DIR := build
DEMO_DIR := demo

# === Include and Library Paths ===
# ODE must be built from source with DrawStuff (Homebrew version doesn't include it)
# Default: /usr/local (where "make install" puts it)
# Override with: make ODE_PREFIX=/path/to/ode
ODE_PREFIX ?= /usr/local
INCLUDES := -I$(ODE_PREFIX)/include -I$(INCLUDE_DIR)
LDFLAGS := -stdlib=libc++ -L$(ODE_PREFIX)/lib
LIBS := -lode -ldrawstuff -lm -framework GLUT -framework OpenGL

# === Texture Path Override ===
# Usage: make TEXTURE_PATH="/path/to/textures"
ifdef TEXTURE_PATH
CXXFLAGS += -DTEXTURE_PATH=\"$(TEXTURE_PATH)\"
endif

# === Source Files ===
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# === Targets ===
TARGET := multisim
DEMO_TARGET := demo_hallway

# === Header Dependencies ===
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)

# === Default Target ===
.PHONY: all clean demo run run-viz help

all: $(TARGET)

# === Main Executable ===
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $(TARGET)"

# === Object Files ===
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# === Build Directory ===
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# === Demo Target ===
demo: $(DEMO_DIR)/demo_hallway.cpp $(filter-out $(BUILD_DIR)/main.o, $(OBJECTS)) | $(BUILD_DIR)
	@echo "Building demo..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(DEMO_TARGET) $^ $(LDFLAGS) $(LIBS)
	@echo "Demo build complete: $(DEMO_TARGET)"

# === Run Targets ===
run: $(TARGET)
	@echo "Starting simulation..."
	./$(TARGET)

run-viz: $(TARGET)
	@echo "Starting simulation with visualization..."
	./scripts/run_simulation.sh

# === Clean ===
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(TARGET) $(DEMO_TARGET)
	@echo "Clean complete."

# === Help ===
help:
	@echo ""
	@echo "MultiSim Build System"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build main simulation"
	@echo "  make demo     - Build simplified hallway demo"
	@echo "  make run      - Build and run simulation"
	@echo "  make run-viz  - Run simulation with Python visualization"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this message"
	@echo ""
	@echo "Options:"
	@echo "  TEXTURE_PATH  - Override ODE texture path"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make TEXTURE_PATH=\"/path/to/ode/drawstuff/textures\""
	@echo "  make run"
	@echo ""
