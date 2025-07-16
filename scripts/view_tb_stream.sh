#!/bin/bash

# Default port value
PORT=47761
HOST="habrokint2"

# Check if a port is provided
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-h|--host hostname] [-d|--dir logdir]"
      exit 1
      ;;
  esac
done

# Create the SSH tunnel
echo "[INFO] Creating SSH tunnel from localhost:6006 to $1:localhost:$PORT"
ssh -L 6006:localhost:$PORT $HOST