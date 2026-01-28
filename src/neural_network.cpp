// neural_network.cpp
// Actor-Critic (A2C) reinforcement learning implementation.
// Uses Adam optimizer with batch gradient accumulation.

#include "neural_network.h"
#include "common.h"
#include "config.h"
#include "csv_export.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <sstream>
#include <string>

using namespace config;

// === Global State ===
double globalWeightChangeBlink = 0.0;
bool globalInitNetworkCalled = false;

// === Network Parameter Arrays ===

// Actor network: Layer 1 (input -> hidden)
static double actor_W1[HIDDEN_SIZE][NUM_INPUTS];
static double actor_b1[HIDDEN_SIZE];
static double actor_W1_m[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double actor_W1_v[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double actor_b1_m[HIDDEN_SIZE] = {0};
static double actor_b1_v[HIDDEN_SIZE] = {0};

// Actor network: Layer 2 (hidden -> output)
static double actor_W2[ACTOR_OUTPUTS][HIDDEN_SIZE];
static double actor_b2[ACTOR_OUTPUTS];
static double actor_W2_m[ACTOR_OUTPUTS][HIDDEN_SIZE] = {0};
static double actor_W2_v[ACTOR_OUTPUTS][HIDDEN_SIZE] = {0};
static double actor_b2_m[ACTOR_OUTPUTS] = {0};
static double actor_b2_v[ACTOR_OUTPUTS] = {0};

// Critic network: Layer 1 (input -> hidden)
static double critic_W1[HIDDEN_SIZE][NUM_INPUTS];
static double critic_b1[HIDDEN_SIZE];
static double critic_W1_m[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double critic_W1_v[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double critic_b1_m[HIDDEN_SIZE] = {0};
static double critic_b1_v[HIDDEN_SIZE] = {0};

// Critic network: Layer 2 (hidden -> output, single value)
static double critic_W2[1][HIDDEN_SIZE];
static double critic_b2[1];
static double critic_W2_m[HIDDEN_SIZE] = {0};
static double critic_W2_v[HIDDEN_SIZE] = {0};
static double critic_b2_m = 0;
static double critic_b2_v = 0;

// === Gradient Accumulators (for batch updates) ===
static double actor_W1_grad[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double actor_b1_grad[HIDDEN_SIZE] = {0};
static double actor_W2_grad[ACTOR_OUTPUTS][HIDDEN_SIZE] = {0};
static double actor_b2_grad[ACTOR_OUTPUTS] = {0};
static double critic_W1_grad[HIDDEN_SIZE][NUM_INPUTS] = {0};
static double critic_b1_grad[HIDDEN_SIZE] = {0};
static double critic_W2_grad[HIDDEN_SIZE] = {0};
static double critic_b2_grad = 0;
static int batch_count = 0;

// Adam optimizer timesteps
static int actor_time = 1;
static int critic_time = 1;

// === Helper Functions ===

/// Adam optimizer update for a single parameter.
static void adam_update(double& param, double grad, double& m, double& v, 
                        int t, double lr) {
    // Clip gradients
    grad = clamp(grad, -GRADIENT_CLIP, GRADIENT_CLIP);
    
    // Update moment estimates
    m = ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (grad * grad);
    
    // Bias correction
    double m_hat = m / (1 - std::pow(ADAM_BETA1, t));
    double v_hat = v / (1 - std::pow(ADAM_BETA2, t));
    
    // Update parameter
    param += lr * m_hat / (std::sqrt(v_hat) + ADAM_EPSILON);
    
    // Clamp parameters to prevent extreme values
    param = clamp(param, -5.0, 5.0);
}

/// Softmax activation function.
static void softmax(const double* z, int n, double* p) {
    double max_z = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > max_z) max_z = z[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        p[i] = std::exp(z[i] - max_z);
        sum += p[i];
    }
    
    for (int i = 0; i < n; i++) {
        p[i] /= sum;
    }
}

/// Compute entropy of a probability distribution.
static double computeEntropy(const double* p, int n) {
    double H = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-6) {
            H -= p[i] * std::log(p[i]);
        }
    }
    return H;
}

// === Public Functions ===

void initNetwork() {
    globalInitNetworkCalled = true;
    
    // Initialize actor network with Xavier initialization
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        actor_b1[i] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        for (int j = 0; j < NUM_INPUTS; j++) {
            actor_W1[i][j] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        }
    }
    
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        actor_b2[k] = xavier_init(HIDDEN_SIZE, ACTOR_OUTPUTS);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            actor_W2[k][i] = xavier_init(HIDDEN_SIZE, ACTOR_OUTPUTS);
        }
    }
    
    // Initialize critic network with Xavier initialization
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_b1[i] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        for (int j = 0; j < NUM_INPUTS; j++) {
            critic_W1[i][j] = xavier_init(NUM_INPUTS, HIDDEN_SIZE);
        }
    }
    
    critic_b2[0] = xavier_init(HIDDEN_SIZE, 1);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_W2[0][i] = xavier_init(HIDDEN_SIZE, 1);
    }
    
    actor_time = 1;
    critic_time = 1;
}

int runNeuralNetwork(Quadruped* quad, double reward, double out_actions[ACTOR_OUTPUTS]) {
    // Clip rewards to reasonable range
    if (reward > -100.0) {
        reward = clamp(reward, -50.0, 50.0);
    } else {
        reward = clamp(reward, -500.0, 50.0);
    }
    
    // === Build Input Vector ===
    double input[NUM_INPUTS];
    
    // Sensor ray values (already in [0,1])
    for (int i = 0; i < NUM_RAYS; i++) {
        input[i] = quad->sensorValues[i];
    }
    
    // Normalized target distance
    input[NUM_RAYS] = clamp(quad->distanceToTarget / 150.0, 0.0, 1.0);
    
    // Normalized position
    input[NUM_RAYS + 1] = (quad->x - WORLD_X_MIN) / WORLD_WIDTH;
    input[NUM_RAYS + 2] = (quad->y - WORLD_Y_MIN) / WORLD_HEIGHT;
    
    // Normalized orientation
    input[NUM_RAYS + 3] = (quad->orientation + M_PI) / (2 * M_PI);
    
    // Relative angle to target (sin/cos encoding)
    double targetAngle = std::atan2(quad->targetY - quad->y, quad->targetX - quad->x);
    double relativeAngle = targetAngle - quad->orientation;
    input[NUM_RAYS + 4] = std::sin(relativeAngle);
    input[NUM_RAYS + 5] = std::cos(relativeAngle);
    
    // Ball visibility features
    input[NUM_RAYS + 6] = quad->ballVisible ? 1.0 : 0.0;
    input[NUM_RAYS + 7] = clamp(quad->ballVisionValue, 0.0, 1.0);
    
    // === Actor Forward Pass ===
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
    
    // === Critic Forward Pass ===
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
    
    // === TD Error Calculation ===
    bool is_terminal = (reward <= -100.0 || reward >= 40.0);
    double td_error;
    
    if (is_terminal) {
        td_error = reward - quad->prevCriticValue;
    } else {
        double td_target = reward + GAMMA * critic_value;
        td_error = td_target - quad->prevCriticValue;
    }
    
    quad->prevCriticValue = critic_value;
    
    // Advantage for actor (includes entropy bonus)
    double actor_advantage = td_error + ENTROPY_COEFF * entropy;
    
    // === Action Selection (Stochastic) ===
    double r = static_cast<double>(std::rand()) / RAND_MAX;
    double cumulative = 0.0;
    int chosen_action = 0;
    for (int i = 0; i < ACTOR_OUTPUTS; i++) {
        cumulative += policy[i];
        if (r <= cumulative) {
            chosen_action = i;
            break;
        }
    }
    
    // Store policy probabilities in output
    for (int i = 0; i < ACTOR_OUTPUTS; i++) {
        out_actions[i] = policy[i];
    }
    
    // === Actor Gradient Computation ===
    double d_actor_z2[ACTOR_OUTPUTS];
    for (int i = 0; i < ACTOR_OUTPUTS; i++) {
        double one_hot = (i == chosen_action) ? 1.0 : 0.0;
        d_actor_z2[i] = (one_hot - policy[i]) * actor_advantage;
    }
    
    // Accumulate actor layer 2 gradients
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        actor_b2_grad[k] += d_actor_z2[k];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            actor_W2_grad[k][i] += d_actor_z2[k] * actor_h1[i];
        }
    }
    
    // Backpropagate to actor layer 1
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
        actor_b1_grad[i] += d_actor_z1[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            actor_W1_grad[i][j] += d_actor_z1[i] * input[j];
        }
    }
    
    // === Critic Gradient Computation ===
    double d_critic = td_error;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_W2_grad[i] += d_critic * critic_h1[i];
    }
    critic_b2_grad += d_critic;
    
    double d_critic_h1[HIDDEN_SIZE] = {0};
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_critic_h1[i] = critic_W2[0][i] * d_critic;
    }
    
    double d_critic_z1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_critic_z1[i] = d_critic_h1[i] * drelu(critic_z1[i]);
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_b1_grad[i] += d_critic_z1[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            critic_W1_grad[i][j] += d_critic_z1[i] * input[j];
        }
    }
    
    batch_count++;
    
    // Trigger visual feedback for large TD errors
    if (std::fabs(td_error) > 10000.0) {
        globalWeightChangeBlink = 1.0;
    }
    
    return chosen_action;
}

void applyBatchUpdate() {
    if (batch_count == 0) return;
    
    double batch_size = static_cast<double>(batch_count);
    
    // Apply actor updates with averaged gradients
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        double avg_grad = actor_b2_grad[k] / batch_size;
        adam_update(actor_b2[k], avg_grad, actor_b2_m[k], actor_b2_v[k], actor_time, ACTOR_LR);
        
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            avg_grad = actor_W2_grad[k][i] / batch_size;
            adam_update(actor_W2[k][i], avg_grad, actor_W2_m[k][i], actor_W2_v[k][i], actor_time, ACTOR_LR);
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double avg_grad = actor_b1_grad[i] / batch_size;
        adam_update(actor_b1[i], avg_grad, actor_b1_m[i], actor_b1_v[i], actor_time, ACTOR_LR);
        
        for (int j = 0; j < NUM_INPUTS; j++) {
            avg_grad = actor_W1_grad[i][j] / batch_size;
            adam_update(actor_W1[i][j], avg_grad, actor_W1_m[i][j], actor_W1_v[i][j], actor_time, ACTOR_LR);
        }
    }
    actor_time++;
    
    // Apply critic updates with averaged gradients
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double avg_grad = critic_W2_grad[i] / batch_size;
        adam_update(critic_W2[0][i], avg_grad, critic_W2_m[i], critic_W2_v[i], critic_time, CRITIC_LR);
    }
    
    double avg_grad = critic_b2_grad / batch_size;
    adam_update(critic_b2[0], avg_grad, critic_b2_m, critic_b2_v, critic_time, CRITIC_LR);
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        avg_grad = critic_b1_grad[i] / batch_size;
        adam_update(critic_b1[i], avg_grad, critic_b1_m[i], critic_b1_v[i], critic_time, CRITIC_LR);
        
        for (int j = 0; j < NUM_INPUTS; j++) {
            avg_grad = critic_W1_grad[i][j] / batch_size;
            adam_update(critic_W1[i][j], avg_grad, critic_W1_m[i][j], critic_W1_v[i][j], critic_time, CRITIC_LR);
        }
    }
    critic_time++;
    
    // Reset gradient accumulators
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        actor_b1_grad[i] = 0;
        critic_b1_grad[i] = 0;
        for (int j = 0; j < NUM_INPUTS; j++) {
            actor_W1_grad[i][j] = 0;
            critic_W1_grad[i][j] = 0;
        }
    }
    
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        actor_b2_grad[k] = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            actor_W2_grad[k][i] = 0;
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        critic_W2_grad[i] = 0;
    }
    critic_b2_grad = 0;
    batch_count = 0;
}

std::string serializeNetworkToCSV(
    const double input[NUM_INPUTS],
    const double actor_z1_arr[HIDDEN_SIZE],
    const double actor_h1_arr[HIDDEN_SIZE],
    const double actor_z2_arr[ACTOR_OUTPUTS],
    const double policy[ACTOR_OUTPUTS],
    int chosen_action,
    double entropy,
    const double critic_z1_arr[HIDDEN_SIZE],
    const double critic_h1_arr[HIDDEN_SIZE],
    double critic_value,
    double reward,
    double td_error,
    double advantage) {
    
    std::ostringstream oss;
    oss << "Category,Layer,Index1,Index2,Value\n";
    
    // Input values
    for (int i = 0; i < NUM_INPUTS; i++) {
        oss << "Input,,," << i << "," << input[i] << "\n";
    }
    
    // Actor intermediate values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        oss << "Actor_z1,Layer1," << i << ",," << actor_z1_arr[i] << "\n";
        oss << "Actor_h1,Layer1," << i << ",," << actor_h1_arr[i] << "\n";
    }
    
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        oss << "Actor_z2,Layer2," << k << ",," << actor_z2_arr[k] << "\n";
        oss << "Policy,Layer2," << k << ",," << policy[k] << "\n";
    }
    
    oss << "Chosen_Action,Output,,," << chosen_action << "\n";
    oss << "Entropy,Output,,," << entropy << "\n";
    
    // Critic intermediate values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        oss << "Critic_z1,Layer1," << i << ",," << critic_z1_arr[i] << "\n";
        oss << "Critic_h1,Layer1," << i << ",," << critic_h1_arr[i] << "\n";
    }
    
    oss << "Critic_Value,Output,,," << critic_value << "\n";
    
    // Learning metrics
    oss << "Reward,,,," << reward << "\n";
    oss << "TD_Error,,,," << td_error << "\n";
    oss << "Advantage,,,," << advantage << "\n";
    
    // Actor weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            oss << "actor_W1,Layer1," << i << "," << j << "," << actor_W1[i][j] << "\n";
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        oss << "actor_b1,Layer1," << i << ",," << actor_b1[i] << "\n";
    }
    
    for (int k = 0; k < ACTOR_OUTPUTS; k++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            oss << "actor_W2,Layer2," << k << "," << i << "," << actor_W2[k][i] << "\n";
        }
        oss << "actor_b2,Layer2," << k << ",," << actor_b2[k] << "\n";
    }
    
    // Critic weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            oss << "critic_W1,Layer1," << i << "," << j << "," << critic_W1[i][j] << "\n";
        }
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        oss << "critic_b1,Layer1," << i << ",," << critic_b1[i] << "\n";
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        oss << "critic_W2,Output,0," << i << "," << critic_W2[0][i] << "\n";
    }
    
    oss << "critic_b2,Output,0,," << critic_b2[0] << "\n";
    
    return oss.str();
}
