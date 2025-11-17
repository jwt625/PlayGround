#!/bin/bash
PID_FILE="logs/server.pid"
[ -f "$PID_FILE" ] && kill $(cat "$PID_FILE") && rm "$PID_FILE" && echo "Server stopped" || echo "No PID file found"

