#ifndef COMMON_H
#define COMMON_H

#include <math.h>
#include <stdlib.h>

// --- Activation functions ---
inline double relu(double x) { return x > 0 ? x : 0; }
inline double drelu(double x) { return x > 0 ? 1.0 : 0.0; }
inline double tanh_activation(double x) { return tanh(x); }

// --- Clamping helper ---
inline double clamp(double val, double minVal, double maxVal) {
    return (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
}

// --- Xavier Initialization ---
inline double xavier_init(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return -limit + ((double)rand() / RAND_MAX) * (2 * limit);
}

#endif // COMMON_H
