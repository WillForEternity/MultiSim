#pragma once

// === Quadruped Agent ===
// Data structure representing a wheeled quadruped robot.
// Contains physics bodies, sensors, and learning state.

#include <ode/ode.h>
#include "config.h"

/// Quadruped agent state.
/// Contains all physics components, sensor data, and reinforcement learning state.
struct Quadruped {
    // === Physics Bodies ===
    dBodyID body;           ///< Main body
    dGeomID bodyGeom;       ///< Body collision geometry
    dBodyID leg[4];         ///< Leg bodies
    dGeomID legGeom[4];     ///< Leg collision geometries
    dJointID hip[4];        ///< Hip joints (body to leg)
    dBodyID wheel[4];       ///< Wheel bodies
    dGeomID wheelTransform[4]; ///< Wheel transform geometries
    dJointID wheelJoint[4]; ///< Wheel joints (leg to wheel)
    
    // === Sensor Rays ===
    dGeomID raySensors[config::NUM_RAYS];   ///< Ray sensor geometries
    double sensorValues[config::NUM_RAYS];  ///< Sensor readings [0,1], 0=close, 1=far
    
    // === Position and Orientation ===
    double x;               ///< Current x position
    double y;               ///< Current y position
    double orientation;     ///< Current yaw angle (radians)
    
    // === Target Tracking ===
    double targetX;         ///< Target x position
    double targetY;         ///< Target y position
    double distanceToTarget;    ///< Current distance to target
    double prevTargetDistance;  ///< Previous distance (for reward shaping)
    
    // === Ball Visibility (from sensors) ===
    bool ballVisible;       ///< True if any ray sensor sees the ball
    double ballVisionValue; ///< Normalized distance to ball via best ray [0,1]
    int ballRayIndex;       ///< Index of ray seeing the ball (-1 if none)
    
    // === Movement Tracking ===
    double spawnX;          ///< X position at spawn
    double spawnTime;       ///< Time of spawn (for timeout)
    double prevX;           ///< Previous x position
    double prevY;           ///< Previous y position
    double lastMoveTime;    ///< Time of last significant movement
    bool stagnancyPunished; ///< Whether stagnation penalty was applied
    
    // === Episode State ===
    double fitness;         ///< Current fitness (x position)
    int respawnCount;       ///< Number of respawns
    double lastRespawnTime; ///< Time of last respawn
    
    // === Collision State ===
    double collisionPenalty; ///< Accumulated collision penalty this step
    
    // === Learning State ===
    double prevCriticValue; ///< Previous critic value for TD learning
    
    // === Visual Feedback ===
    int blinkType;          ///< Type of blink (BLINK_NONE, GREEN, RED, BLUE)
    int blinkCount;         ///< Number of blink cycles remaining
    double blinkStartTime;  ///< Start time of blink animation
};
