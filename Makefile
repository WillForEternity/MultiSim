CXX = g++-14
CXXFLAGS = -stdlib=libc++ -I/usr/local/include -fopenmp -O2 -Wall
LDFLAGS = -L/usr/local/lib -lode -ldrawstuff -lm -framework GLUT -framework OpenGL
SRC = src/environment.cpp src/neural_network.cpp
TARGET = multiSim2

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
