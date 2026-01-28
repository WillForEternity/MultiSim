#pragma once

// === Common Utilities ===
// Math helper functions for neural networks and physics.

#include <cmath>
#include <cstdlib>

/// ReLU activation function.
/// @param x Input value
/// @return max(0, x)
inline double relu(double x) {
    return x > 0 ? x : 0;
}

/// ReLU derivative.
/// @param x Input value (pre-activation)
/// @return 1 if x > 0, else 0
inline double drelu(double x) {
    return x > 0 ? 1.0 : 0.0;
}

/// Tanh activation function.
/// @param x Input value
/// @return tanh(x)
inline double tanh_activation(double x) {
    return std::tanh(x);
}

/// Clamp a value to a range.
/// @param val Value to clamp
/// @param minVal Minimum allowed value
/// @param maxVal Maximum allowed value
/// @return Clamped value
inline double clamp(double val, double minVal, double maxVal) {
    return (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
}

/// Xavier/Glorot weight initialization.
/// Draws from uniform distribution [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
/// @param fan_in Number of input connections
/// @param fan_out Number of output connections
/// @return Initialized weight value
inline double xavier_init(int fan_in, int fan_out) {
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    return -limit + (static_cast<double>(std::rand()) / RAND_MAX) * (2 * limit);
}

/// Normalize an angle to [-PI, PI].
/// @param angle Angle in radians
/// @return Normalized angle
inline double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

/// Generate a random double in range [min, max].
/// @param min Minimum value
/// @param max Maximum value
/// @return Random value in range
inline double randomUniform(double min, double max) {
    return min + (static_cast<double>(std::rand()) / RAND_MAX) * (max - min);
}
