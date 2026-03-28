#!/bin/bash

source ~/scripts/activate_venv.sh

# Default port value
PORT=47761
DIR="~/scratch/runs/"

# Check if a port is provided
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -d|--dir)
      DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-h|--host hostname] [-d|--dir logdir]"
      exit 1
      ;;
  esac
done

# Start tensorboard with the specified port
echo "[INFO] Starting TensorBoard on port $PORT"
tensorboard --logdir $DIR --port $PORT --bind_all
