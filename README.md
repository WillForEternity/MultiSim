# MultiSim

**MultiSim** is a simulation framework that integrates a physics-based engine with an actor–critic reinforcement learning algorithm to enable wheeled quadrupeds to learn navigation in dynamic obstacle courses. The project is implemented in C++ using the Open Dynamics Engine (ODE) for physics simulation and visualization, while the neural networks are built from scratch with an actor–critic architecture.

![MultiSim](Intro.png)

---

# Development in progress...

## Overview

MultiSim’s primary goal is to train wheeled quadrupeds to navigate complex, ever-changing environments. The system comprises two core components:

- **Physics-Based Simulation:**  
  Uses ODE to simulate realistic rigid-body dynamics, collision detection, ray casting, and environmental interactions. The simulation creates a dynamic obstacle course with boundary walls, a target object, and thousands of randomly placed obstacles. Each quadruped is modeled as a composite of a central body, legs, and wheels. A grid of ray "sensors" attached to the body provides distance measurements for obstacle detection.

- **Actor–Critic Reinforcement Learning:**  
  The learning algorithm is based on an actor–critic framework. The **actor network** outputs one of four possible control actions (e.g., different wheel velocities for forward, turning, or backwards movement), while the **critic network** estimates the value of the current state. Temporal-difference (TD) learning updates both networks using gradients computed via the Adam optimizer and utilizes Polyak averaging for stable target updates.

---

## Setup and Run

Follow these steps to build and run MultiSim on your machine:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/WillForEternity/MultiSim.git
   cd MultiSim

2. **Install Dependencies**
Ensure you have:
- A C++14-compliant compiler (e.g., g++-14)
- The Open Dynamics Engine (https://www.ode.org, https://bitbucket.org/odedevs/ode/downloads/) 
- OpenGL and GLUT for graphics

3. **Compile and Run**
You can compile using the provided Makefile or directly with:
```bash
g++-14 -stdlib=libc++ -I/usr/local/include -L/usr/local/lib -o multiSim2 environment.cpp neural_network.cpp main.cpp -lode -ldrawstuff -lm -framework GLUT -framework OpenGL -fopenmp
```
This compile command worked for me (m3 Macbook Air) but some fiddling may be needed. Adjust the include and library paths if necessary. To run the simulation, execute the compiled binary, of course:
```bash
./multiSim2
```
A simulation window will open displaying the dynamic obstacle course and the quadrupeds. The neural networks will begin training in real time as the quadrupeds navigate toward the target.

4. **Adjusting Parameters**
- Hyperparameters: Modify learning rates, discount factors, and entropy regularization in neural_network.cpp.
- Simulation Settings: Adjust the number of quadrupeds, sensor grid size, and obstacle density in environment.cpp.

Enjoy! 

