#!/bin/bash

# Default port value
PORT=47761
HOST="habrokint2"
LOCAL_PORT=6006

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
    -l|--localport)
      LOCAL_PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-h|--host hostname] [-d|--dir logdir] [-l|--localport local_port]"
      exit 1
      ;;
  esac
done

# Create the SSH tunnel
echo "[INFO] Creating SSH tunnel from localhost:6006 to $1:localhost:$PORT"
ssh -L $LOCAL_PORT:localhost:$PORT $HOST
