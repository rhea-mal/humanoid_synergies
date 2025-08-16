#!/bin/bash

# Path to your executables
EXECUTABLE="toro_controller"
PLOT_EXECUTABLE="../optitrack/graphcs/plot_metrics.py"

# Launch each executable in the background with arguments
$EXECUTABLE --name Hannah &
$EXECUTABLE --name Tracy &
$PLOT_EXECUTABLE &

# Wait for all background processes to finish (optional)
wait