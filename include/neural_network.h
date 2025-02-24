#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "quadruped.h"
#include <string>

// -----------------------------------------------------------------------------
// Network settings, hyperparameters, and Adam parameters
#define NUM_RAYS          32
#define NUM_INPUTS        (NUM_RAYS + 4)   // sensor rays + target distance + x, y, orientation
#define ACTOR_OUTPUTS     4                // 4 discrete commands
#define HIDDEN_SIZE       64

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

// -----------------------------------------------------------------------------
// C-linkage functions (C-compatible functions)
#ifdef __cplusplus
extern "C" {
#endif

void initNetwork();
void runNeuralNetwork(Quadruped *quad, double reward, double out_actions[ACTOR_OUTPUTS]);

#ifdef __cplusplus
}  // end extern "C"
#endif

// C++ only function declarations:
#ifdef __cplusplus
std::string serializeNetworkToCSV(
    const double input[NUM_INPUTS],
    const double actor_z1[HIDDEN_SIZE],
    const double actor_h1[HIDDEN_SIZE],
    const double actor_z2[ACTOR_OUTPUTS],
    const double policy[ACTOR_OUTPUTS],
    int chosen_action,
    double entropy,
    const double critic_z1[HIDDEN_SIZE],
    const double critic_h1[HIDDEN_SIZE],
    double critic_value,
    double reward,
    double td_error,
    double advantage
);
#endif

#endif // NEURAL_NETWORK_H
