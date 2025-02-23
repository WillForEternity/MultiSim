#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "quadruped.h"  
#include "common.h"     // For macros like SENSOR_RAY_GRID, etc.

// --- Network settings ---
// Increase the number of rays for full circle sensing.
#define NUM_RAYS          32
#define NUM_INPUTS        (NUM_RAYS + 4)   // sensor rays + target distance + x and y coordinate + orientation
#define ACTOR_OUTPUTS     4                // 4 discrete wheel commands
#define HIDDEN_SIZE       32

extern const double ACTOR_LR;
extern const double CRITIC_LR;
extern const double GAMMA;
extern const double ALPHA_ENTROPY;
extern const double POLYAK;

struct AdamParams {
    double lr;
    double beta1;
    double beta2;
    double epsilon;
};
extern const struct AdamParams adamDefault;

#ifdef __cplusplus
extern "C" {
#endif

void initNetwork();
void runNeuralNetwork(Quadruped *quad, double reward, double out_actions[ACTOR_OUTPUTS]);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H
