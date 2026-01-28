# MultiSim

**MultiSim** is a simulation framework that integrates a physics-based engine with an Actor-Critic (A2C) reinforcement learning algorithm to enable wheeled quadrupeds to learn navigation in dynamic obstacle courses. The project is implemented in C++17 using the Open Dynamics Engine (ODE) for physics simulation and visualization, with neural networks built from scratch.

![MultiSim](Intro.png)

## Features

- **Physics-Based Simulation**: Uses ODE for realistic rigid-body dynamics, collision detection, ray casting, and environmental interactions
- **Actor-Critic Reinforcement Learning**: Custom A2C implementation with separate actor and critic networks
- **Parallel Training**: Multiple quadrupeds learn simultaneously with synchronized batch gradient updates
- **Real-Time Visualization**: Watch agents learn in real-time with sensor ray visualization and reward feedback
- **Visual Feedback**: Agents blink colors (green=reward, red=penalty, blue=weight change) to indicate learning events
- **Configurable**: All parameters centralized in `include/config.h` for easy tuning

## Quick Start

```bash
# Clone and build
git clone https://github.com/WillForEternity/MultiSim.git
cd MultiSim
make

# Run simulation
./multisim
```

## Requirements

- **C++17 compiler** (clang++ recommended on macOS)
- **Open Dynamics Engine (ODE)** with DrawStuff library
- **OpenGL and GLUT** (comes with macOS)
- **Python 3** with matplotlib and numpy (optional, for visualization script)

### Installing ODE on macOS

**Important:** The Homebrew version of ODE does *not* include DrawStuff (the visualization library). You must build ODE from source:

```bash
# Download ODE source
cd ~/Downloads
curl -LO https://bitbucket.org/odedevs/ode/downloads/ode-0.16.6.tar.gz
tar xzf ode-0.16.6.tar.gz
cd ode-0.16.6

# Configure with DrawStuff for macOS
./configure --enable-double-precision --with-drawstuff=OSX --prefix=/usr/local

# Build and install
make -j4
sudo make install
```

If you installed to a different prefix, specify it when building:
```bash
make ODE_PREFIX=/your/ode/path
```

## Build Targets

```bash
make            # Build main simulation
make demo       # Build simplified hallway demo
make run        # Build and run simulation
make run-viz    # Run with Python visualization
make clean      # Remove build artifacts
make help       # Show all options
```

## Texture Path Configuration

If textures are not found at runtime, specify the path when building:

```bash
make TEXTURE_PATH="/path/to/ode/drawstuff/textures"
```

Common locations:
- `/usr/local/share/ode/drawstuff/textures`
- `/opt/homebrew/share/ode/drawstuff/textures`
- `~/Downloads/ode-0.16.x/drawstuff/textures`

## Controls

| Key | Action |
|-----|--------|
| `p` | Pause/resume simulation |
| `q` | Quit |
| Mouse drag | Rotate camera |
| Mouse scroll | Zoom |

## Project Structure

```
MultiSim/
├── include/               # Header files
│   ├── config.h           # All configuration constants
│   ├── common.h           # Math utilities (clamp, relu, xavier_init)
│   ├── quadruped.h        # Agent data structure
│   ├── neural_network.h   # A2C interface
│   ├── environment.h      # Simulation environment
│   ├── physics.h          # ODE wrappers
│   ├── rendering.h        # Visualization
│   └── csv_export.h       # Data export
├── src/                   # Implementation files
│   ├── main.cpp           # Entry point
│   ├── environment.cpp    # Simulation logic
│   ├── neural_network.cpp # A2C implementation with Adam optimizer
│   ├── physics.cpp        # Body/joint creation
│   ├── rendering.cpp      # Drawing code
│   └── csv_export.cpp     # CSV writing
├── demo/                  # Simplified demos
│   └── demo_hallway.cpp   # 1D navigation demo
├── scripts/               # Utility scripts
│   ├── run_simulation.sh  # Run with visualization
│   └── visualize_network.py # Network visualization
├── Makefile               # Build system
└── README.md
```

## Configuration

Edit `include/config.h` to modify:

### Simulation Settings
- `NUM_QUADRUPEDS` - Number of parallel agents (default: 10)
- `NUM_OBSTACLES` - Obstacle count (default: 2000)
- `SIMULATION_DT` - Physics timestep (default: 0.015s)
- `EPISODE_TIMEOUT` - Max episode length in seconds (default: 10.0s)

### Neural Network Architecture
- `NUM_RAYS` - Sensor rays per agent (default: 32)
- `NUM_FEATURES` - Additional state features (default: 8)
- `NUM_INPUTS` - Total input size: NUM_RAYS + NUM_FEATURES (default: 40)
- `HIDDEN_SIZE` - Hidden layer neurons (default: 128)
- `ACTOR_OUTPUTS` - Number of actions (default: 4)

### Learning Hyperparameters
- `ACTOR_LR` - Actor learning rate (default: 0.0003)
- `CRITIC_LR` - Critic learning rate (default: 0.001)
- `GAMMA` - Discount factor (default: 0.99)
- `ENTROPY_COEFF` - Exploration bonus (default: 0.001)
- `GRADIENT_CLIP` - Max gradient magnitude (default: 1.0)

### Rewards
- `GOAL_REWARD` - Reaching target (default: +50)
- `TIMEOUT_PENALTY` - Episode timeout (default: -50)
- `FALLING_PENALTY` - Agent falls over (default: -50)
- `COLLISION_PENALTY` - Obstacle collision (default: -2)
- `TIME_PENALTY` - Living cost per step (default: -0.05)
- `DISTANCE_SCALE` - Multiplier for distance improvement reward (default: 20)

### World Bounds
- `WORLD_X_MIN/MAX` - X bounds (default: -80 to 60)
- `WORLD_Y_MIN/MAX` - Y bounds (default: -70 to 70)
- `TARGET_X/Y` - Goal position (default: 55, 0)
- `GOAL_DISTANCE` - Distance to trigger goal reached (default: 4.0)

## Architecture

The neural network uses a two-headed Actor-Critic architecture with shared input processing:

```
Input Layer (40 neurons)
├── 32 ray sensor distances [0,1] (0=obstacle close, 1=clear)
├── Normalized target distance
├── Normalized position (x, y)
├── Normalized orientation [0,1]
├── Relative angle to target (sin, cos encoding)
├── Ball visible flag (0 or 1)
└── Ball vision value [0,1]

Actor Network                    Critic Network
├── Hidden: 128 neurons (ReLU)   ├── Hidden: 128 neurons (ReLU)
└── Output: 4 actions (softmax)  └── Output: 1 value estimate
```

### Actions
1. **Forward** - All wheels move forward at `WHEEL_VEL_FORWARD`
2. **Turn Left** - Differential steering (left slow, right fast)
3. **Turn Right** - Differential steering (left fast, right slow)
4. **Backward** - All wheels move backward at `WHEEL_VEL_BACK`

### Agent Morphology

Each quadruped consists of:
- **Body**: Central rigid body (0.4m × 0.15m × 0.05m)
- **Legs**: 4 legs with hip joints connecting to body
- **Wheels**: 4 wheels with motor joints for locomotion
- **Sensors**: 32 ray sensors in a cone (±30°) for obstacle detection

## Visualization

Run the network visualization script alongside the simulation:

```bash
./scripts/run_simulation.sh
```

Or manually:
```bash
./multisim &
python3 scripts/visualize_network.py
```

This displays:
- Network weight visualizations for actor and critic
- Policy distribution bar charts
- Real-time learning metrics (TD error, entropy, rewards)

## Troubleshooting

### Build Errors

**"ode/ode.h not found"**
```bash
# Make sure ODE is installed and headers are in include path
export CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH
```

**"library not found for -lode"**
```bash
# Add library path
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
```

### Runtime Issues

**"Could not load texture"**
- Set `TEXTURE_PATH` at build time (see above)

**Simulation crashes immediately**
- Check that ODE was built with DrawStuff support
- Verify OpenGL is working: `glxinfo | head -3` (Linux) or run a simple OpenGL app (macOS)

**Agents don't learn**
- Increase `NUM_QUADRUPEDS` for more training data
- Reduce `ENTROPY_COEFF` if exploration is too high
- Check reward values in console output
- Ensure `EPISODE_TIMEOUT` gives agents enough time to reach the goal

## Technical Details

### Batch Learning
All agents share the same neural network. Gradients are accumulated from each agent and averaged before applying a single synchronized update via `applyBatchUpdate()`. This provides stable learning with diverse experience.

### TD Learning
Uses temporal difference (TD) learning with the update rule:
```
δ = r + γV(s') - V(s)
```
Terminal states (goal reached, timeout, fallen) use the reward directly without bootstrapping.

### Adam Optimizer
Both actor and critic use Adam optimization with:
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8
- Gradient clipping at ±1.0
- Parameter clamping at ±5.0 to prevent extreme values

### Weight Initialization
Xavier initialization is used for all network weights, ensuring proper gradient flow at the start of training.

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- [Open Dynamics Engine](https://www.ode.org/) for physics simulation
- [OpenAI Baselines](https://github.com/openai/baselines) for A2C reference implementation
