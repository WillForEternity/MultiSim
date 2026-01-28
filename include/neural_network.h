#pragma once

// === Neural Network ===
// Actor-Critic (A2C) reinforcement learning implementation.
// Features:
// - Actor network: Outputs policy (action probabilities)
// - Critic network: Estimates state value
// - Adam optimizer with gradient clipping
// - Batch gradient accumulation across all agents

#include "quadruped.h"
#include "config.h"
#include <string>

/// Initialize network weights using Xavier initialization.
/// Must be called before any other neural network functions.
void initNetwork();

/// Run forward and backward pass for a single agent.
/// Accumulates gradients for later batch update.
/// @param quad Agent state (sensors, position, orientation)
/// @param reward Reward signal for TD learning
/// @param outActions Output buffer for policy probabilities (size: ACTOR_OUTPUTS)
/// @return Index of chosen action (stochastic sampling from policy)
int runNeuralNetwork(Quadruped* quad, double reward, 
                     double outActions[config::ACTOR_OUTPUTS]);

/// Apply accumulated gradients from all agents (synchronized batch update).
/// Should be called once per simulation step after all agents have been processed.
/// Uses Adam optimizer with gradient averaging across the batch.
void applyBatchUpdate();

/// Global flag indicating significant weight change (triggers visual feedback).
extern double globalWeightChangeBlink;

/// Global flag indicating network has been initialized.
extern bool globalInitNetworkCalled;

/// Serialize network state to CSV for visualization.
/// Exports input values, intermediate activations, weights, and learning metrics.
/// @param input Input vector (size: NUM_INPUTS)
/// @param actor_z1 Actor hidden layer pre-activation (size: HIDDEN_SIZE)
/// @param actor_h1 Actor hidden layer activation (size: HIDDEN_SIZE)
/// @param actor_z2 Actor output pre-activation (size: ACTOR_OUTPUTS)
/// @param policy Policy probabilities (size: ACTOR_OUTPUTS)
/// @param chosen_action Selected action index
/// @param entropy Policy entropy
/// @param critic_z1 Critic hidden layer pre-activation (size: HIDDEN_SIZE)
/// @param critic_h1 Critic hidden layer activation (size: HIDDEN_SIZE)
/// @param critic_value Critic value estimate
/// @param reward Current reward
/// @param td_error Temporal difference error
/// @param advantage Advantage estimate
/// @return CSV-formatted string
std::string serializeNetworkToCSV(
    const double input[config::NUM_INPUTS],
    const double actor_z1[config::HIDDEN_SIZE],
    const double actor_h1[config::HIDDEN_SIZE],
    const double actor_z2[config::ACTOR_OUTPUTS],
    const double policy[config::ACTOR_OUTPUTS],
    int chosen_action,
    double entropy,
    const double critic_z1[config::HIDDEN_SIZE],
    const double critic_h1[config::HIDDEN_SIZE],
    double critic_value,
    double reward,
    double td_error,
    double advantage
);
