#!/bin/bash
# run_simulation.sh
#
# This script runs the MultiSim simulation in the background
# and then launches the Python visualization script.
#
# Usage: ./scripts/run_simulation.sh
#
# The visualization will read network_parameters.csv as it's updated.
# When you close the Python window or press Ctrl+C, the simulation stops.

set -e

# Change to project root
cd "$(dirname "$0")/.."

echo "Building MultiSim..."
make -j4

echo ""
echo "Starting simulation..."
./multisim &
SIM_PID=$!

echo "Simulation started (PID $SIM_PID)"
echo ""

# Wait for simulation to initialize and create CSV data
sleep 2

# Trap Ctrl+C to kill simulation when visualization closes
trap "echo 'Stopping simulation...'; kill $SIM_PID 2>/dev/null; exit 0" INT TERM

# Launch visualization
echo "Starting visualization..."
echo "Close the Python window or press Ctrl+C to stop."
echo ""
python3 scripts/visualize_network.py

# Kill simulation when visualization exits
kill $SIM_PID 2>/dev/null
echo "Done."
