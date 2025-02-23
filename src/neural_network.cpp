#include "../include/neural_network.h"
#include "../include/common.h"
#include "../include/quadruped.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- Hyperparameters ---
const double ACTOR_LR = 0.05;
const double CRITIC_LR = 0.05;
const double GAMMA = 0.9;
const double ALPHA_ENTROPY = 0.05;
const double POLYAK = 0.05;

// --- Adam parameters ---
const AdamParams adamDefault = {0.001, 0.9, 0.999, 1e-8};

// Global blink trigger variables.
double globalWeightChangeBlink = 0.0;
bool globalInitNetworkCalled = false;

// --- Network parameter arrays ---
// Actor network: Layer 1
static double actor_W1[HIDDEN_SIZE][NUM_INPUTS];
static double actor_b1[HIDDEN_SIZE];
// Adam moment estimates for actor_W1 and actor_b1:
static double actor_W1_m[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double actor_W1_v[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double actor_b1_m[HIDDEN_SIZE] = {0};
static double actor_b1_v[HIDDEN_SIZE] = {0};

// Actor network: Layer 2
static double actor_W2[ACTOR_OUTPUTS][HIDDEN_SIZE];
static double actor_b2[ACTOR_OUTPUTS];
static double actor_W2_m[ACTOR_OUTPUTS][HIDDEN_SIZE] = {0};
static double actor_W2_v[ACTOR_OUTPUTS][HIDDEN_SIZE] = {0};
static double actor_b2_m[ACTOR_OUTPUTS] = {0};
static double actor_b2_v[ACTOR_OUTPUTS] = {0};

// Critic network: Layer 1
static double critic_W1[HIDDEN_SIZE][NUM_INPUTS];
static double critic_b1[HIDDEN_SIZE];
static double critic_W1_m[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double critic_W1_v[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double critic_b1_m[HIDDEN_SIZE] = {0};
static double critic_b1_v[HIDDEN_SIZE] = {0};

// Critic network: Layer 2 (output)
static double critic_W2[1][HIDDEN_SIZE];
static double critic_b2[1];
static double critic_W2_m[HIDDEN_SIZE] = {0};
static double critic_W2_v[HIDDEN_SIZE] = {0};
static double critic_b2_m = 0;
static double critic_b2_v = 0;

// Target critic network.
static double critic_W1_target[HIDDEN_SIZE][NUM_INPUTS];
static double critic_b1_target[HIDDEN_SIZE];
static double critic_W2_target[1][HIDDEN_SIZE];
static double critic_b2_target[1];

// Global time steps for Adam.
static int actor_time = 1;
static int critic_time = 1;

// --- Helper: Adam update for a single parameter ---
// Note: This function uses C++ references, so it must not be wrapped in extern "C".
static void adam_update(double &param, double grad, double &m, double &v, int t, const AdamParams &adam) {
    m = adam.beta1 * m + (1 - adam.beta1) * grad;
    v = adam.beta2 * v + (1 - adam.beta2) * (grad * grad);
    double m_hat = m / (1 - pow(adam.beta1, t));
    double v_hat = v / (1 - pow(adam.beta2, t));
    param += adam.lr * m_hat / (sqrt(v_hat) + adam.epsilon);
    param = clamp(param, -2.0, 2.0);
}

// --- Xavier Initialization for all parameters ---
void initNetwork() {
    globalInitNetworkCalled = true;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        actor_b1[i] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        critic_b1[i] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        for (int j = 0; j < NUM_INPUTS; j++) {
            actor_W1[i][j] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
            critic_W1[i][j] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
            critic_W1_target[i][j] = critic_W1[i][j];
        }
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
        critic_W2_target[0][i] = critic_W2[0][i];
    }
    critic_b2_target[0] = critic_b2[0];
    
    actor_time = 1;
    critic_time = 1;
}

// --- Polyak update for target critic ---
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

// --- Softmax function ---
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

// --- Compute entropy of a probability vector ---
static double computeEntropy(const double *p, int n) {
    double H = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-6)
            H -= p[i] * log(p[i]);
    }
    return H;
}

/******************************************************************************
 * runNeuralNetwork: Run the actor–critic update.
 ******************************************************************************
 * This function implements the forward and backward passes for both the
 * actor and critic networks and updates their weights using the Adam
 * optimizer.
 */
void runNeuralNetwork(Quadruped *quad, double reward, double out_actions[ACTOR_OUTPUTS])
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
    double policy[ACTOR_OUTPUTS];
    softmax(actor_z2, ACTOR_OUTPUTS, policy);
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
    
    // --- Critic forward pass (target) ---
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
    
    // --- TD error and advantage ---
    double td_error = reward + GAMMA * critic_target_value - quad->prevCriticValue;
    double advantage = td_error + ALPHA_ENTROPY * entropy;
    
    // --- Action selection: sample from policy distribution ---
    double r = (double)rand() / RAND_MAX;
    double cumulative = 0.0;
    int chosen_action = 0;
    for (int i = 0; i < ACTOR_OUTPUTS; i++) {
        cumulative += policy[i];
        if (r <= cumulative) {
            chosen_action = i;
            break;
        }
    }

    for (int i = 0; i < ACTOR_OUTPUTS; i++) {
        out_actions[i] = policy[i];
    }
    
    // --- Compute gradients for actor using softmax derivative ---
    double d_actor_z2[ACTOR_OUTPUTS];
    for (int i = 0; i < ACTOR_OUTPUTS; i++) {
        double one_hot = (i == chosen_action) ? 1.0 : 0.0;
        d_actor_z2[i] = (one_hot - policy[i]) * advantage;
    }
    
    // --- Backpropagate actor: update actor_W2 and actor_b2 ---
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        adam_update(actor_b2[k], d_actor_z2[k], actor_b2_m[k], actor_b2_v[k], actor_time, adamDefault);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double grad = d_actor_z2[k] * actor_h1[i];
            adam_update(actor_W2[k][i], grad, actor_W2_m[k][i], actor_W2_v[k][i], actor_time, adamDefault);
        }
    }
    
    // --- Backpropagate to layer 1 for actor ---
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
        adam_update(actor_b1[i], d_actor_z1[i], actor_b1_m[i], actor_b1_v[i], actor_time, adamDefault);
        for (int j = 0; j < NUM_INPUTS; j++) {
            double grad = d_actor_z1[i] * input[j];
            adam_update(actor_W1[i][j], grad, actor_W1_m[i][j], actor_W1_v[i][j], actor_time, adamDefault);
        }
    }
    actor_time++;
    
    // --- Critic update ---
    double d_critic = td_error;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double grad = d_critic * critic_h1[i];
        adam_update(critic_W2[0][i], grad, critic_W2_m[i], critic_W2_v[i], critic_time, adamDefault);
    }
    adam_update(critic_b2[0], d_critic, critic_b2_m, critic_b2_v, critic_time, adamDefault);
    
    double d_critic_h1[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_critic_h1[i] = critic_W2[0][i] * d_critic;
    }
    double d_critic_z1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_critic_z1[i] = d_critic_h1[i] * drelu(critic_z1[i]);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        adam_update(critic_b1[i], d_critic_z1[i], critic_b1_m[i], critic_b1_v[i], critic_time, adamDefault);
        for (int j = 0; j < NUM_INPUTS; j++) {
            double grad = d_critic_z1[i] * input[j];
            adam_update(critic_W1[i][j], grad, critic_W1_m[i][j], critic_W1_v[i][j], critic_time, adamDefault);
        }
    }
    critic_time++;
    
    // --- Update stored critic value ---
    quad->prevCriticValue = critic_value;
    
    // --- Polyak update for target critic ---
    polyakUpdate();
    
    // --- Global weight change blink trigger ---
    if (fabs(td_error) > 5000.0)
        globalWeightChangeBlink = 1.0;
}
