// main.cpp
// MultiSim Entry Point
// Reinforcement learning simulation for wheeled quadrupeds.

#include "environment.h"
#include "neural_network.h"
#include "config.h"

#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Create and initialize environment
    Environment env;
    env.initialize();
    env.createObstacles(config::NUM_OBSTACLES);
    
    // Initialize neural network
    initNetwork();
    
    // Spawn agents
    env.spawnAgents(config::NUM_QUADRUPEDS);
    
    // Get DrawStuff callbacks and run simulation
    dsFunctions callbacks = env.getCallbacks();
    
    // Handle potential empty argv
    int dummyArgc = 1;
    char* dummyArgv[] = {const_cast<char*>("multisim"), nullptr};
    if (argc == 0) {
        argc = dummyArgc;
        argv = dummyArgv;
    }
    
    dsSimulationLoop(argc, argv, 800, 600, &callbacks);
    
    // Cleanup
    env.cleanup();
    
    return 0;
}
