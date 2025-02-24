#!/bin/bash
# run_all.sh
#
# This script runs the DrawStuff simulation in the background
# and then launches the Python visualization script.
#
# It waits a few seconds to let the simulation start writing the CSV file,
# and when the Python script exits, it kills the simulation.

# Start the simulation in the background
./multiSim2 &
SIM_PID=$!

echo "Started simulation (PID $SIM_PID)"

# Wait a few seconds for the simulation to initialize and create CSV data
sleep 2

# Launch the Python visualization script
python3 visualize_network.py

echo "Visualization script finished. Terminating simulation..."

# Kill the simulation process when the visualization script is closed
kill $SIM_PID
