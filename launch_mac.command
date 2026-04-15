#!/bin/bash

# Move to the directory where this script is located (critical for USB drives)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================="
echo "     Starting Identity Shift App (Auto)"
echo "=========================================="

# 1. Kill old processes blocking port 8080
if lsof -Ti:8080 > /dev/null; then
    echo "Killing existing process on port 8080..."
    lsof -ti:8080 | xargs kill -9 2>/dev/null
fi

# 2. Launch Bootstrapper in Background
echo "[INFO] Launching Python Server..."
python3 start_server.py &
SERVER_PID=$!

# Wait for server to boot up
sleep 3

# 3. Open the browser to the auto-start URL to skip selection
echo "[INFO] Opening Interface..."
open "http://localhost:8080/autostart"

# Wait for process so the terminal stays open
wait $SERVER_PID
