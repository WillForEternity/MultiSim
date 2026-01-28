#pragma once

// === Environment ===
// Main simulation environment managing the ODE world and quadruped agents.

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>
#include "quadruped.h"
#include "config.h"

/// Environment class managing the simulation world and agents.
class Environment {
public:
    Environment();
    ~Environment();
    
    /// Initialize ODE world, boundaries, and target.
    void initialize();
    
    /// Create random obstacles in the environment.
    /// @param count Number of obstacles to create
    void createObstacles(int count);
    
    /// Spawn quadruped agents at starting positions.
    /// @param count Number of agents to spawn
    void spawnAgents(int count);
    
    /// Get DrawStuff callback functions for simulation loop.
    dsFunctions getCallbacks();
    
    /// Clean up all ODE resources.
    void cleanup();
    
    // Accessors for simulation state
    double getSimulationTime() const { return simulationTime_; }
    Quadruped* getQuadrupeds() { return quads_; }
    int getQuadrupedCount() const { return numQuadrupeds_; }

private:
    // ODE world state
    dWorldID world_;
    dSpaceID space_;
    dJointGroupID contactGroup_;
    dGeomID groundPlane_;
    dGeomID boundaryBoxes_[4];
    
    // Target
    dBodyID targetBall_;
    dGeomID targetBallGeom_;
    
    // Obstacles
    dGeomID obstacles_[config::MAX_OBSTACLES];
    dVector3 obstacleSizes_[config::MAX_OBSTACLES];
    int numObstacles_;
    
    // Agents
    Quadruped quads_[config::NUM_QUADRUPEDS];
    int numQuadrupeds_;
    
    // Simulation time
    double simulationTime_;
    
    // Callback functions (static to work with C API)
    static void simLoopCallback(int pause);
    static void startCallback();
    static void commandCallback(int cmd);
    static void stopCallback();
    
    // Collision handling
    static void nearCallback(void* data, dGeomID o1, dGeomID o2);
    
    // Internal methods
    void updateSensorsAndControl(Quadruped* quad);
    void replaceQuadruped(int index);
    bool hasFallen(Quadruped* quad);
    double getEnvironmentHeight(double x, double y);
    
    // Static instance pointer for callbacks
    static Environment* instance_;
};
