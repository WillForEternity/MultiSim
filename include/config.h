#pragma once

// === MultiSim Configuration ===
// All simulation constants in one place for easy tuning.
// Uses C++17 inline constexpr for type safety and ODR compliance.

#include <cmath>

namespace config {

// === Simulation Settings ===
inline constexpr int    NUM_QUADRUPEDS    = 10;
inline constexpr double SIMULATION_DT     = 0.015;
inline constexpr int    NUM_OBSTACLES     = 2000;
inline constexpr int    MAX_OBSTACLES     = 10000;
inline constexpr double EPISODE_TIMEOUT   = 10.0;  // seconds

// === Neural Network Architecture ===
inline constexpr int    NUM_RAYS          = 32;
inline constexpr int    NUM_FEATURES      = 8;   // target dist, x, y, orientation, sin/cos angle, ball visibility/value
inline constexpr int    NUM_INPUTS        = NUM_RAYS + NUM_FEATURES;  // 40 total
inline constexpr int    HIDDEN_SIZE       = 128;
inline constexpr int    ACTOR_OUTPUTS     = 4;

// === Hyperparameters (validated against A2C best practices) ===
inline constexpr double ACTOR_LR          = 0.0003;  // Lower than critic for stability
inline constexpr double CRITIC_LR         = 0.001;   // Higher to learn values faster
inline constexpr double GAMMA             = 0.99;    // Standard discount factor
inline constexpr double ENTROPY_COEFF     = 0.001;   // Exploration bonus
inline constexpr double GRADIENT_CLIP     = 1.0;     // Prevent exploding gradients

// === Adam Optimizer (standard defaults) ===
inline constexpr double ADAM_BETA1        = 0.9;
inline constexpr double ADAM_BETA2        = 0.999;
inline constexpr double ADAM_EPSILON      = 1e-8;

// === Physics/Geometry ===
inline constexpr double BODY_LENGTH       = 0.4;
inline constexpr double BODY_WIDTH        = 0.15;
inline constexpr double BODY_HEIGHT       = 0.05;
inline constexpr double BODY_MASS         = 10.0;
inline constexpr double LEG_WIDTH         = 0.05;
inline constexpr double LEG_LENGTH        = 0.3;
inline constexpr double LEG_MASS          = 8.0;
inline constexpr double WHEEL_RADIUS      = 0.1;
inline constexpr double WHEEL_WIDTH       = 0.1;
inline constexpr double WHEEL_MASS        = 8.0;
inline constexpr double HIP_FMAX          = 1000.0;
inline constexpr double WHEEL_FMAX        = 1000.0;
inline constexpr double SENSOR_MAX_DIST   = 3.0;
inline constexpr double DRAWN_MAX_LENGTH  = 3.0;
inline constexpr double SENSOR_CONE_ANGLE = M_PI / 6.0;

// === World Bounds ===
inline constexpr double WORLD_X_MIN       = -80.0;
inline constexpr double WORLD_X_MAX       = 60.0;
inline constexpr double WORLD_Y_MIN       = -70.0;
inline constexpr double WORLD_Y_MAX       = 70.0;
inline constexpr double WORLD_WIDTH       = WORLD_X_MAX - WORLD_X_MIN;  // 140
inline constexpr double WORLD_HEIGHT      = WORLD_Y_MAX - WORLD_Y_MIN;  // 140

// === Target Ball ===
inline constexpr double TARGET_X          = 55.0;
inline constexpr double TARGET_Y          = 0.0;
inline constexpr double TARGET_RADIUS     = 2.0;
inline constexpr double GOAL_DISTANCE     = 4.0;  // Distance to consider goal reached

// === Rewards ===
inline constexpr double GOAL_REWARD       = 50.0;
inline constexpr double TIMEOUT_PENALTY   = -50.0;
inline constexpr double FALLING_PENALTY   = -50.0;
inline constexpr double DISTANCE_SCALE    = 20.0;   // Multiplier for distance improvement reward
inline constexpr double COLLISION_PENALTY = -2.0;   // Light collision penalty
inline constexpr double LEG_COLLISION_PENALTY = -0.5;
inline constexpr double WALL_COLLISION_PENALTY = -1.0;
inline constexpr double TIME_PENALTY      = -0.05;  // Living cost to encourage efficiency

// === Behavior Thresholds ===
inline constexpr double FALLING_THRESHOLD = 0.3;    // up_z below this = fallen
inline constexpr double STAGNATION_TIME   = 20000.0;
inline constexpr double SIDE_WALL_THRESHOLD = 0.3;

// === Visual Feedback ===
inline constexpr int    BLINK_NONE        = 0;
inline constexpr int    BLINK_BLUE        = 1;   // Weight change
inline constexpr int    BLINK_GREEN       = 2;   // Reward
inline constexpr int    BLINK_RED         = 3;   // Punishment
inline constexpr double BLINK_PERIOD      = 0.2;

// === Wheel Velocities ===
inline constexpr double WHEEL_VEL_FORWARD = -40.0;
inline constexpr double WHEEL_VEL_BACK    = 30.0;
inline constexpr double WHEEL_VEL_TURN_FAST = -40.0;
inline constexpr double WHEEL_VEL_TURN_SLOW = 10.0;

} // namespace config

// === Texture Path (allow compile-time override) ===
#ifndef TEXTURE_PATH
#define TEXTURE_PATH "/Users/williamnorden/Downloads/ode-0.16.6/drawstuff/textures"
#endif
