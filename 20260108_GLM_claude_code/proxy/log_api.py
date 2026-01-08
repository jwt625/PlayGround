#!/usr/bin/env python3
"""
Simple API server to serve log files to the viewer.
"""

import json
import os
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
API_PORT = int(os.getenv("API_PORT", "58736"))


def read_all_logs():
    """Read all JSONL log files and return as list."""
    logs = []
    
    if not LOG_DIR.exists():
        return logs
    
    # Get all .jsonl files sorted by modification time (newest first)
    log_files = sorted(
        LOG_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    # Sort by timestamp (newest first)
    logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return logs


@app.route('/api/logs')
def get_logs():
    """Return all logs as JSON."""
    logs = read_all_logs()
    return jsonify(logs)


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "log_dir": str(LOG_DIR.absolute()),
        "log_count": len(read_all_logs())
    })


if __name__ == "__main__":
    print(f"Starting log API server on port {API_PORT}")
    print(f"Reading logs from: {LOG_DIR.absolute()}")
    app.run(host="127.0.0.1", port=API_PORT, debug=False)

