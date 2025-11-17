#!/bin/bash
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/server_${TIMESTAMP}.log"
PID_FILE="logs/server.pid"
nohup uv run python server.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Server started. PID: $(cat $PID_FILE). Log: $LOG_FILE"

