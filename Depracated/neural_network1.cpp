// neural_network.cpp
// Compile with: (see environment.cpp)

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

// Network settings.
#define SENSOR_RAY_GRID   5
#define NUM_RAYS          (SENSOR_RAY_GRID * SENSOR_RAY_GRID)
#define NUM_INPUTS        (NUM_RAYS+1)  // sensor rays + target distance.
#define ACTOR_OUTPUTS     4             // 4 discrete wheel commands.
#define HIDDEN_SIZE       8
const double ACTOR_SIGMA = 0.1;
const double ACTOR_LR = 0.01;
const double CRITIC_LR = 0.02;
const double GAMMA = 0.9;
const double ALPHA_ENTROPY = 0.01; // entropy bonus coefficient

// For the target network update.
const double POLYAK = 0.01;

// Global blink trigger variables.
double globalWeightChangeBlink = 0.0;
bool globalInitNetworkCalled = false;

// The Quadruped structure for neural network is assumed to have:
//   sensorValues[NUM_RAYS], distanceToTarget, and prevCriticValue.
typedef struct Quadruped {
    double sensorValues[NUM_RAYS];
    double distanceToTarget;
    double prevCriticValue;  // For TD bootstrapping.
} Quadruped;

// Clamp helper.
static double clamp(double val, double minVal, double maxVal) {
    if(val < minVal) return minVal;
    if(val > maxVal) return maxVal;
    return val;
}

// --- Xavier Initialization ---
static double xavier_init(int fan_in, int fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return -limit + ((double)rand()/RAND_MAX) * (2*limit);
}

// Actor network parameters.
static double actor_W1[HIDDEN_SIZE][NUM_INPUTS];
static double actor_b1[HIDDEN_SIZE];
static double actor_W2[ACTOR_OUTPUTS][HIDDEN_SIZE];
static double actor_b2[ACTOR_OUTPUTS];

// Critic network parameters.
static double critic_W1[HIDDEN_SIZE][NUM_INPUTS];
static double critic_b1[HIDDEN_SIZE];
static double critic_W2[1][HIDDEN_SIZE];
static double critic_b2[1];

// Target critic parameters.
static double critic_W1_target[HIDDEN_SIZE][NUM_INPUTS];
static double critic_b1_target[HIDDEN_SIZE];
static double critic_W2_target[1][HIDDEN_SIZE];
static double critic_b2_target[1];

static double relu(double x) { return x > 0 ? x : 0; }
static double drelu(double x) { return x > 0 ? 1.0 : 0.0; }
static double tanh_activation(double x) { return tanh(x); }

// Compute softmax of a vector.
static void softmax(const double *z, int n, double *p) {
    double max_z = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > max_z) max_z = z[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        p[i] = exp(z[i] - max_z);
        sum += p[i];
    }
    for (int i = 0; i < n; i++) {
        p[i] /= sum;
    }
}

// Compute entropy given probability vector.
static double computeEntropy(const double *p, int n) {
    double H = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-6)
            H -= p[i] * log(p[i]);
    }
    return H;
}

/******************************************************************************
 * initNetwork: Initialize network parameters (using Xavier init) and copy critic to target.
 ******************************************************************************/
void initNetwork() {
    globalInitNetworkCalled = true;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        actor_b1[i] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        critic_b1[i] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        for (int j = 0; j < NUM_INPUTS; j++) {
            actor_W1[i][j] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
            critic_W1[i][j] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
            // Also initialize target critic weights.
            critic_W1_target[i][j] = critic_W1[i][j];
        }
        // Copy biases for critic target.
        critic_b1_target[i] = critic_b1[i];
    }
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        actor_b2[k] = xavier_init(HIDDEN_SIZE, ACTOR_OUTPUTS);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            actor_W2[k][i] = xavier_init(HIDDEN_SIZE, ACTOR_OUTPUTS);
        }
    }
    critic_b2[0] = xavier_init(HIDDEN_SIZE, 1);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_W2[0][i] = xavier_init(HIDDEN_SIZE, 1);
        // Copy to target.
        critic_W2_target[0][i] = critic_W2[0][i];
    }
    critic_b2_target[0] = critic_b2[0];
}

/******************************************************************************
 * polyakUpdate: Update target critic parameters with Polyak averaging.
 ******************************************************************************/
static void polyakUpdate() {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            critic_W1_target[i][j] = POLYAK * critic_W1[i][j] + (1 - POLYAK) * critic_W1_target[i][j];
        }
        critic_b1_target[i] = POLYAK * critic_b1[i] + (1 - POLYAK) * critic_b1_target[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_W2_target[0][i] = POLYAK * critic_W2[0][i] + (1 - POLYAK) * critic_W2_target[0][i];
    }
    critic_b2_target[0] = POLYAK * critic_b2[0] + (1 - POLYAK) * critic_b2_target[0];
}

/******************************************************************************
 * runNeuralNetwork: Run the actor–critic update.
 * It computes:
 *  - Actor forward pass with softmax to obtain probabilities and entropy.
 *  - Critic forward pass using both online and target networks.
 *  - TD error: delta = reward + gamma * V_target(s') - V(s)
 *  - Then uses (delta + ALPHA_ENTROPY * entropy) as the advantage for actor updates.
 *  - Finally, updates network parameters via SGD (replace with Adam if desired) and performs Polyak update.
 ******************************************************************************/
void runNeuralNetwork(struct Quadruped *quad, double reward, double out_actions[ACTOR_OUTPUTS])
{
    double input[NUM_INPUTS];
    for (int i = 0; i < NUM_RAYS; i++) {
        input[i] = quad->sensorValues[i];
    }
    input[NUM_RAYS] = quad->distanceToTarget;
    
    // --- Actor forward pass ---
    double actor_z1[HIDDEN_SIZE], actor_h1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        actor_z1[i] = actor_b1[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            actor_z1[i] += actor_W1[i][j] * input[j];
        }
        actor_h1[i] = relu(actor_z1[i]);
    }
    double actor_z2[ACTOR_OUTPUTS];
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        actor_z2[k] = actor_b2[k];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            actor_z2[k] += actor_W2[k][i] * actor_h1[i];
        }
    }
    // Compute policy probabilities via softmax.
    double policy[ACTOR_OUTPUTS];
    softmax(actor_z2, ACTOR_OUTPUTS, policy);
    // Compute entropy.
    double entropy = computeEntropy(policy, ACTOR_OUTPUTS);
    
    // --- Critic forward pass (online) ---
    double critic_z1[HIDDEN_SIZE], critic_h1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_z1[i] = critic_b1[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            critic_z1[i] += critic_W1[i][j] * input[j];
        }
        critic_h1[i] = relu(critic_z1[i]);
    }
    double critic_value = critic_b2[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_value += critic_W2[0][i] * critic_h1[i];
    }
    
    // --- Critic forward pass (target) for next state:
    // For simplicity, assume the same input is used (in real use, use next state).
    double critic_target_z1[HIDDEN_SIZE], critic_target_h1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_target_z1[i] = critic_b1_target[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            critic_target_z1[i] += critic_W1_target[i][j] * input[j];
        }
        critic_target_h1[i] = relu(critic_target_z1[i]);
    }
    double critic_target_value = critic_b2_target[0];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_target_value += critic_W2_target[0][i] * critic_target_h1[i];
    }
    
    // --- TD error computation using target network ---
    double td_error = reward + GAMMA * critic_target_value - quad->prevCriticValue;
    // Effective advantage adds an entropy bonus.
    double advantage = td_error + ALPHA_ENTROPY * entropy;
    
    // --- Actor update (simple SGD update) ---
    // Compute gradient for actor_W2 and actor_b2.
    double d_actor_z2[ACTOR_OUTPUTS];
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        // Derivative of softmax is complex; for simplicity, use (output - probability)
        // Here we assume a simple gradient: proportional to advantage.
        d_actor_z2[k] = advantage; // simplified
        // Update actor biases.
        actor_b2[k] += ACTOR_LR * d_actor_z2[k];
        actor_b2[k] = clamp(actor_b2[k], -2.0, 2.0);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double deltaW = ACTOR_LR * d_actor_z2[k] * actor_h1[i];
            actor_W2[k][i] += deltaW;
            actor_W2[k][i] = clamp(actor_W2[k][i], -2.0, 2.0);
        }
    }
    // Similarly update actor_W1 and actor_b1.
    double d_actor_h1[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int k = 0; k < ACTOR_OUTPUTS; k++) {
            d_actor_h1[i] += actor_W2[k][i] * d_actor_z2[k];
        }
    }
    double d_actor_z1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_actor_z1[i] = d_actor_h1[i] * drelu(actor_z1[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        actor_b1[i] += ACTOR_LR * d_actor_z1[i];
        actor_b1[i] = clamp(actor_b1[i], -2.0, 2.0);
        for (int j = 0; j < NUM_INPUTS; j++) {
            double deltaW = ACTOR_LR * d_actor_z1[i] * input[j];
            actor_W1[i][j] += deltaW;
            actor_W1[i][j] = clamp(actor_W1[i][j], -2.0, 2.0);
        }
    }
    
    // --- Critic update (simple SGD update) ---
    double d_critic_z1[HIDDEN_SIZE];
    double d_critic_h1[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_critic_h1[i] = critic_W2[0][i] * td_error;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_critic_z1[i] = d_critic_h1[i] * drelu(critic_z1[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_b1[i] += CRITIC_LR * d_critic_z1[i];
        critic_b1[i] = clamp(critic_b1[i], -2.0, 2.0);
        for (int j = 0; j < NUM_INPUTS; j++) {
            double deltaW = CRITIC_LR * d_critic_z1[i] * input[j];
            critic_W1[i][j] += deltaW;
            critic_W1[i][j] = clamp(critic_W1[i][j], -2.0, 2.0);
        }
    }
    critic_b2[0] += CRITIC_LR * td_error;
    critic_b2[0] = clamp(critic_b2[0], -2.0, 2.0);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double deltaW = CRITIC_LR * td_error * critic_h1[i];
        critic_W2[0][i] += deltaW;
        critic_W2[0][i] = clamp(critic_W2[0][i], -2.0, 2.0);
    }
    
    if (fabs(td_error) > 0.1)
        globalWeightChangeBlink = 1.0;
    
    // --- Update the stored critic value for next step ---
    quad->prevCriticValue = critic_value;
    
    // --- Polyak update for target critic ---
    polyakUpdate();
}

