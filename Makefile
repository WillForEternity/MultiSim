CXX = clang++
CXXFLAGS = -stdlib=libc++ -I/usr/local/include -O2 -Wall
LDFLAGS = -stdlib=libc++ -L/usr/local/lib -lode -ldrawstuff -lm -framework GLUT -framework OpenGL

SRC = src/environment.cpp src/neural_network.cpp src/socket.cpp
TARGET = multiSim2

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
