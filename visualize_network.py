#!/usr/bin/env python3
"""
This script reads dynamically updated "network_parameters.csv" and displays two concurrent windows:

Window 1 (Neural Networks):
  - Actor Network diagram (left)
  - Critic Network diagram (right)
  A horizontal colorbar below the diagrams shows a gradient (jet cmap) mapping
  from low (negative high weight) to high (positive high weight).

Window 2 (Other Parameters):
  - Actor outputs & Policy (top-left)
  - Actor Info (top-right)
  - Critic & Learning Metrics (bottom row, spanning both columns)

Both windows update concurrently every 2 seconds.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec, colors
from matplotlib.colorbar import ColorbarBase

# --- Network dimensions ---
NUM_INPUTS     = 36
HIDDEN_SIZE    = 64
ACTOR_OUTPUTS  = 4
CRITIC_OUTPUTS = 1

def read_csv_data(filename):
    """
    Reads the CSV file and returns a dictionary with keys for each category.
    For categories with indices (e.g., "Input", "Actor_z1", etc.), stores values in a dict.
    For singular categories (with no indices), stores the value directly.
    """
    data = {}
    categories = ["Input", "Actor_z1", "Actor_h1", "Actor_z2", "Policy",
                  "Chosen_Action", "Entropy",
                  "Critic_z1", "Critic_h1", "Critic_Value",
                  "Reward", "TD_Error", "Advantage",
                  "actor_W1", "actor_b1", "actor_W2", "actor_b2",
                  "critic_W1", "critic_b1", "critic_W2", "critic_b2"]
    for cat in categories:
        data[cat] = {}
    try:
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cat = row["Category"]
                if cat not in data:
                    data[cat] = {}
                val_str = row.get("Value")
                if not val_str:
                    continue
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    continue
                idx1 = row["Index1"]
                idx2 = row["Index2"]
                if idx1 == "" and idx2 == "":
                    data[cat] = val
                elif idx2 == "":
                    try:
                        i = int(idx1)
                    except:
                        continue
                    data[cat][i] = val
                else:
                    try:
                        i = int(idx1)
                        j = int(idx2)
                    except:
                        continue
                    data[cat][(i, j)] = val
    except FileNotFoundError:
        return None
    return data

def get_matrix(matrix_dict):
    """
    Converts a dictionary of weight values (keys: (i,j)) into a list of tuples:
      (source_index, target_index, weight).
    We swap (i, j) so the diagram draws connections left-to-right.
    """
    matrix = []
    for (i, j), weight in matrix_dict.items():
        matrix.append((j, i, weight))
    return matrix

def draw_network(ax, input_data, weight1, weight2, title, output_size):
    """
    Draws a three-layer network diagram:
      - Input layer (left), Hidden layer (middle), Output layer (right).
      - Input nodes are colored by value (viridis).
      - Edges use jet colormap, thickness ~ magnitude.
    """
    ax.clear()
    ax.set_title(title, fontsize=16, pad=10)
    ax.axis("off")
    
    # Node positions
    input_positions  = [(0, y) for y in np.linspace(1, -1, NUM_INPUTS)]
    hidden_positions = [(1, y) for y in np.linspace(1, -1, HIDDEN_SIZE)]
    output_positions = [(2, y) for y in np.linspace(1, -1, output_size)]
    
    # Draw input nodes (viridis colormap)
    if len(input_data) > 0:
        input_vals = np.array(input_data)
        norm_input = (input_vals - np.min(input_vals)) / (np.ptp(input_vals) + 1e-6)
    else:
        norm_input = np.zeros(NUM_INPUTS)
    for i, pos in enumerate(input_positions):
        color = plt.cm.viridis(norm_input[i])
        ax.scatter(pos[0], pos[1], s=120, color=color, zorder=3)
    
    # Hidden and output nodes
    for pos in hidden_positions:
        ax.scatter(pos[0], pos[1], s=120, color="lightblue", zorder=3)
    for pos in output_positions:
        ax.scatter(pos[0], pos[1], s=120, color="orange", zorder=3)
    
    # Edges: from input to hidden
    cmap = plt.get_cmap("jet")
    weights1 = [abs(w) for (_, _, w) in weight1]
    max_w1 = max(weights1) if weights1 else 1.0
    for (src, tgt, w) in weight1:
        if src < NUM_INPUTS and tgt < HIDDEN_SIZE:
            start = input_positions[src]
            end = hidden_positions[tgt]
            norm_val = (w + max_w1) / (2 * max_w1)
            color = cmap(norm_val)
            lw = 0.5 + 3 * abs(w) / max_w1
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, zorder=1)
    
    # Edges: from hidden to output
    weights2 = [abs(w) for (_, _, w) in weight2]
    max_w2 = max(weights2) if weights2 else 1.0
    for (src, tgt, w) in weight2:
        if src < HIDDEN_SIZE and tgt < output_size:
            start = hidden_positions[src]
            end = output_positions[tgt]
            norm_val = (w + max_w2) / (2 * max_w2)
            color = cmap(norm_val)
            lw = 0.5 + 3 * abs(w) / max_w2
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, zorder=1)
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.5, 1.5)

def draw_gradient_legend(fig):
    """
    Adds a horizontal colorbar at the bottom of the figure to serve as a gradient legend.
    The colorbar uses 'jet' from 0 (Low) to 1 (High).
    """
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.03])  # [left, bottom, width, height]
    norm = colors.Normalize(vmin=0, vmax=1)
    cb = ColorbarBase(cbar_ax, cmap=plt.get_cmap("jet"), norm=norm, orientation='horizontal')
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "0", "High"])

def draw_actor_output(ax, actor_z2, policy):
    """
    Displays a grouped bar chart for actor outputs and policy probabilities.
    """
    ax.clear()
    ax.set_title("Actor Outputs & Policy", fontsize=14, pad=10)
    if actor_z2 and policy:
        indices = sorted(actor_z2.keys())
        z2_vals = [actor_z2[i] for i in indices]
        policy_vals = [policy[i] for i in indices]
        width = 0.35
        ax.bar([x - width/2 for x in indices], z2_vals, width, label="Actor_z2", color="skyblue")
        ax.bar([x + width/2 for x in indices], policy_vals, width, label="Policy", color="salmon")
        ax.set_xlabel("Output Index")
        ax.set_ylabel("Value")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "No Actor Output Data", ha="center", va="center", fontsize=14)
    ax.set_xlim(-0.5, len(actor_z2.keys()) + 0.5 if actor_z2 else 3)

def draw_text_info(ax, info_dict, title):
    """
    Displays key-value pairs from info_dict as a formatted multi-line text,
    with the given title at the top of the subplot.
    """
    ax.clear()
    ax.set_title(title, fontsize=14, pad=10)
    lines = []
    for key, value in info_dict.items():
        if isinstance(value, float):
            lines.append(f"{key:<15}: {value:>10.4f}")
        else:
            lines.append(f"{key:<15}: {value}")
    text_str = "\n".join(lines)
    ax.text(0.05, 0.5, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', family='monospace')
    ax.axis("off")

def update_all(frame):
    filename = "network_parameters.csv"
    data = read_csv_data(filename)
    if data is None:
        return

    # Extract input vector
    input_data = [data["Input"].get(i, 0) for i in range(NUM_INPUTS)]
    
    # Actor outputs & policy
    actor_z2  = data["Actor_z2"]
    policy    = data["Policy"]
    chosen_action = data["Chosen_Action"] if isinstance(data["Chosen_Action"], float) else None
    entropy   = data["Entropy"] if isinstance(data["Entropy"], float) else None

    # Critic & learning metrics
    critic_value = data["Critic_Value"] if isinstance(data["Critic_Value"], float) else None
    reward    = data["Reward"] if isinstance(data["Reward"], float) else None
    td_error  = data["TD_Error"] if isinstance(data["TD_Error"], float) else None
    advantage = data["Advantage"] if isinstance(data["Advantage"], float) else None

    # Weight matrices
    actor_W1 = get_matrix(data["actor_W1"])
    actor_W2 = get_matrix(data["actor_W2"])
    critic_W1 = get_matrix(data["critic_W1"])
    critic_W2 = get_matrix(data["critic_W2"])
    
    # --- Update NN Window ---
    draw_network(ax_actor_net, input_data, actor_W1, actor_W2, "Actor Network", ACTOR_OUTPUTS)
    draw_network(ax_critic_net, input_data, critic_W1, critic_W2, "Critic Network", CRITIC_OUTPUTS)
    
    # --- Update Other Params Window ---
    draw_actor_output(ax_actor_out, actor_z2, policy)
    
    # Actor info
    actor_info = {}
    if chosen_action is not None:
        actor_info["Chosen Action"] = int(chosen_action)
    if entropy is not None:
        actor_info["Entropy"] = entropy
    draw_text_info(ax_actor_info, actor_info, "Actor Info")
    
    # Critic & learning metrics
    critic_info = {}
    if critic_value is not None:
        critic_info["Critic Value"] = critic_value
    if reward is not None:
        critic_info["Reward"] = reward
    if td_error is not None:
        critic_info["TD Error"] = td_error
    if advantage is not None:
        critic_info["Advantage"] = advantage
    draw_text_info(ax_critic_info, critic_info, "Critic & Learning Metrics")

# === Figure 1: Neural Network Diagrams ===
fig_network = plt.figure(figsize=(16, 8))
gs_net = gridspec.GridSpec(1, 2)
ax_actor_net = fig_network.add_subplot(gs_net[0, 0])
ax_critic_net = fig_network.add_subplot(gs_net[0, 1])
fig_network.suptitle("Neural Network Diagrams", fontsize=18)
draw_gradient_legend(fig_network)  # horizontal colorbar at the bottom

# === Figure 2: Other Parameters ===
fig_other = plt.figure(figsize=(16, 8))
gs_other = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# Top row
ax_actor_out  = fig_other.add_subplot(gs_other[0, 0])
ax_actor_info = fig_other.add_subplot(gs_other[0, 1])

# Bottom row: single subplot spanning columns
ax_critic_info = fig_other.add_subplot(gs_other[1, :])

fig_other.suptitle("Other Parameters", fontsize=18)
plt.tight_layout()

# === Animations (both windows update together) ===
ani_network = FuncAnimation(fig_network, update_all, interval=2000, cache_frame_data=False)
ani_other   = FuncAnimation(fig_other, update_all, interval=2000, cache_frame_data=False)

plt.show()