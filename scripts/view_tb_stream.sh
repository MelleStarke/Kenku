#!/bin/bash

# Default port value
PORT=47761

# Check if a port is provided as second argument
if [ "$2" = "--port" ] && [ ! -z "$3" ]; then
  PORT=$3
fi

# Create the SSH tunnel
echo "[INFO] Creating SSH tunnel from localhost:6006 to $1:localhost:$PORT"
ssh -L 6006:localhost:$PORT "$1"