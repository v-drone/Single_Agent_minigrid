#!/bin/bash

echo "Starting Docker container for AirSim..."
cmd.exe /C "docker start $1"
echo "Starting AirSim server on configuration: $1"
python airsim_client/airsim_runner.py -f "$1" &
