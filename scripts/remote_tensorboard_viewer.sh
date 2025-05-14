#!/bin/bash

# Script to manage remote tensorboard streaming and local viewing
# With automatic cleanup on script termination

# Default port range to try
PORT_MIN=47000
PORT_MAX=48000

# Parse command line arguments
REMOTE_HOST="habrokint2"
PORT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -h|--host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-h|--host hostname] [-p|--port port]"
      exit 1
      ;;
  esac
done

# Remote server details
VENV_SCRIPT="~/scripts/activate_venv.sh"
TB_STREAM_SCRIPT="~/scripts/stream_tensorboard.sh"
LOCAL_VIEW_SCRIPT="$HOME/Kenku/scripts/view_tb_stream.sh"

# Temporary file to store PIDs
TMP_PIDFILE="/tmp/tensorboard_session_$$.pid"

# Function to clean up processes on exit
cleanup() {
    echo -e "\n[INFO] Shutting down tensorboard session..."
    
    # If we have stored the local viewer PID, kill it
    if [ -f "$TMP_PIDFILE" ]; then
        LOCAL_PID=$(cat "$TMP_PIDFILE")
        if ps -p $LOCAL_PID > /dev/null; then
            echo "[INFO] Stopping local tensorboard viewer (PID: $LOCAL_PID)"
            kill $LOCAL_PID 2>/dev/null
        fi
        rm -f "$TMP_PIDFILE"
    fi
    
    # Terminate the SSH process which should also kill the remote tensorboard stream
    if [ ! -z "$SSH_PID" ]; then
        echo "[INFO] Stopping remote tensorboard stream (SSH PID: $SSH_PID)"
        kill $SSH_PID 2>/dev/null
    fi
    
    echo "[INFO] Cleanup complete"
}

# Function to find an available port on the remote server
find_available_port() {
    if [ ! -z "$PORT" ]; then
        echo "$PORT"
        return
    fi
    
    echo "[INFO] Finding available port on remote server..."
    # Try ports in our range until we find one that's available
    FOUND_PORT=$(ssh $REMOTE_HOST "for p in \$(seq $PORT_MIN $PORT_MAX); do nc -z localhost \$p || { echo \$p; exit 0; }; done")
    
    if [ -z "$FOUND_PORT" ]; then
        echo "[ERROR] Could not find available port in range $PORT_MIN-$PORT_MAX" >&2
        exit 1
    fi
    
    echo "[INFO] Found available port: $FOUND_PORT"
    echo "$FOUND_PORT"
}

# Register the cleanup function to be called on script exit
trap cleanup EXIT INT TERM

echo "[INFO] Starting tensorboard session"

# Start the remote tensorboard stream in the background
echo "[INFO] Starting remote tensorboard stream..."
ssh -T $REMOTE_HOST "source $VENV_SCRIPT && $TB_STREAM_SCRIPT --port $PORT" &
SSH_PID=$!

# Give the stream a moment to initialize
sleep 2

# Start the local viewer in the background
echo "[INFO] Starting local tensorboard viewer..."
$LOCAL_VIEW_SCRIPT $REMOTE_HOST "--port $PORT" &
LOCAL_PID=$!

# Store the local viewer PID for cleanup
echo $LOCAL_PID > "$TMP_PIDFILE"

echo "[INFO] Tensorboard session active"
echo "[INFO] Press Ctrl+C to terminate both processes and clean up"

# Wait for the SSH process to complete (or be killed)
wait $SSH_PID

# The cleanup function will be called automatically when the script exits