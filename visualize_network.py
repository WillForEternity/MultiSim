#!/usr/bin/env python3
"""
visualize_network_dynamic.py

A script to parse "network_parameters.csv" (generated by the C++ simulation)
and visualize the weight matrices for the actor and critic networks in a single window
with subplots. The visualization updates dynamically to reflect new network parameters.

Usage:
  python3 visualize_network_dynamic.py
"""

import csv
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Helper: HSV to RGB conversion ---
def hsv2rgb(h, s, v):
    """
    Convert an HSV color (h in [0,360], s and v in [0,1]) to RGB (each in [0,1]).
    """
    c = v * s
    h_prime = (h / 60.0) % 6
    x = c * (1 - abs((h_prime % 2) - 1))
    m = v - c

    if 0 <= h_prime < 1:
        r1, g1, b1 = c, x, 0
    elif 1 <= h_prime < 2:
        r1, g1, b1 = x, c, 0
    elif 2 <= h_prime < 3:
        r1, g1, b1 = 0, c, x
    elif 3 <= h_prime < 4:
        r1, g1, b1 = 0, x, c
    elif 4 <= h_prime < 5:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x

    return (r1 + m, g1 + m, b1 + m)

# --- Visualization for a single weight matrix ---
def visualize_matrix(ax, matrix, layer_name, source_count, target_count):
    """
    Visualize a weight matrix on the provided Axes object.
    
    Parameters:
      - ax: matplotlib Axes to draw on.
      - matrix: list of tuples (source_index, target_index, weight)
      - layer_name: string (e.g., "actor_W1")
      - source_count: number of neurons in the source layer
      - target_count: number of neurons in the target layer
    """
    # Clear previous content on the axis
    ax.clear()
    ax.set_title(layer_name)
    ax.axis('off')

    if not matrix:
        ax.text(0.5, 0.5, "No connections", ha='center', va='center')
        return

    # Determine maximum absolute weight for normalization
    max_weight = max(abs(w) for (_, _, w) in matrix)

    # Arrange nodes in two vertical columns:
    #   Source nodes at x=0, target nodes at x=1.
    # Distribute nodes vertically and center them around y=0.
    src_positions = [i - (source_count - 1) / 2 for i in range(source_count)]
    tgt_positions = [i - (target_count - 1) / 2 for i in range(target_count)]

    # Draw source nodes (left column)
    for i, y in enumerate(src_positions):
        ax.plot(0, y, 'ko', markersize=8)
        ax.text(-0.1, y, f"{i}", ha='right', va='center', fontsize=8)

    # Draw target nodes (right column)
    for j, y in enumerate(tgt_positions):
        ax.plot(1, y, 'ko', markersize=8)
        ax.text(1.1, y, f"{j}", ha='left', va='center', fontsize=8)

    # Draw connections between nodes
    for (src_idx, tgt_idx, weight) in matrix:
        norm = abs(weight) / max_weight if max_weight != 0 else 0
        linewidth = 0.5 + 4 * norm

        # Set color based on sign: negative weights are red (hue=0), positive are blue (hue=240)
        hue = 0 if weight < 0 else 240
        saturation = norm
        value = 1.0
        color = hsv2rgb(hue, saturation, value)

        # Start and end positions
        x_start, y_start = 0, src_positions[src_idx]
        x_end, y_end = 1, tgt_positions[tgt_idx]
        ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=linewidth)

    ax.set_xlim(-0.5, 1.5)
    # Set limits to include all nodes plus some margin.
    ymin = min(src_positions + tgt_positions) - 1
    ymax = max(src_positions + tgt_positions) + 1
    ax.set_ylim(ymin, ymax)

def read_csv_matrices(filename, dims):
    """
    Read CSV file and return a dictionary mapping layer names to weight matrices.
    Each matrix is a list of tuples (source_index, target_index, weight).
    """
    matrices = {layer: [] for layer in dims.keys()}

    try:
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if not row["Value"]:
                    continue
                try:
                    value = float(row["Value"])
                except ValueError:
                    continue

                layer = row["Layer"]
                index1 = row["Index1"]
                index2 = row["Index2"]

                # Skip bias entries (where index2 is empty)
                if index2 == "":
                    continue

                i = int(index1)
                j = int(index2)
                # The CSV is stored as row=target, col=source. Swap them for left-to-right.
                if layer in matrices:
                    matrices[layer].append((j, i, value))
    except FileNotFoundError:
        # If file not found, simply return empty matrices.
        pass

    return matrices

def update(frame):
    """
    Update function called by FuncAnimation.
    Reads the CSV file, updates each subplot with the latest weight matrix.
    """
    filename = "network_parameters.csv"
    # Expected dimensions for each weight matrix
    dims = {
        "actor_W1":   (36, 64),  # NUM_INPUTS=36 => (32 rays + 4 states), hidden=64
        "actor_W2":   (64, 4),   # hidden=64 => 4 discrete outputs
        "critic_W1":  (36, 64),  # same input size for critic, hidden=64
        "critic_W2":  (64, 1)    # hidden=64 => single critic output
    }

    matrices = read_csv_matrices(filename, dims)
    
    # For each subplot, update the visualization
    for ax, (layer_name, (src_count, tgt_count)) in zip(axes.flat, dims.items()):
        matrix = matrices.get(layer_name, [])
        visualize_matrix(ax, matrix, layer_name, src_count, tgt_count)

# Create a single figure with 2x2 subplots for each network weight matrix
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plt.tight_layout()

# Set up animation to update every 2000 ms (2 seconds)
ani = FuncAnimation(fig, update, interval=2000)

plt.show()
