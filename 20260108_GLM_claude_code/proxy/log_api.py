#!/usr/bin/env python3
"""
Simple API server to serve log files to the viewer.
"""

import json
import os
import pickle
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from log_classifier import enrich_logs

load_dotenv()

app = Flask(__name__)
CORS(app)

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
API_PORT = int(os.getenv("API_PORT", "58736"))
CACHE_FILE = LOG_DIR / ".enriched_cache.pkl"

# Cache for enriched data
_cache = {
    'logs': None,
    'enriched_data': None,
    'last_modified': None
}


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
    
    # Sort by timestamp (oldest first for workflow graph processing)
    # Frontend can reverse if needed for display
    logs.sort(key=lambda x: x.get('timestamp', ''))
    
    return logs


def get_latest_log_mtime():
    """Get the latest modification time of all log files."""
    if not LOG_DIR.exists():
        return None

    log_files = list(LOG_DIR.glob("*.jsonl"))
    if not log_files:
        return None

    return max(f.stat().st_mtime for f in log_files)


def load_cache_from_disk():
    """Load cached enriched data from disk."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, 'rb') as f:
            cached = pickle.load(f)
            print(f"Loaded cache from disk: {len(cached.get('logs', []))} logs")
            return cached
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


def save_cache_to_disk(cache_data):
    """Save enriched data cache to disk."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved cache to disk: {CACHE_FILE}")
    except Exception as e:
        print(f"Failed to save cache: {e}")


@app.route('/api/logs')
def get_logs():
    """Return all logs as JSON with enriched metadata and workflow graph."""
    latest_mtime = get_latest_log_mtime()

    # Try memory cache first
    if (_cache['enriched_data'] is not None and
        _cache['last_modified'] is not None and
        latest_mtime is not None and
        latest_mtime <= _cache['last_modified']):
        print("Using memory cache")
        return jsonify(_cache['enriched_data'])

    # Try disk cache
    if latest_mtime is not None and CACHE_FILE.exists():
        cache_mtime = CACHE_FILE.stat().st_mtime
        if cache_mtime >= latest_mtime:
            disk_cache = load_cache_from_disk()
            if disk_cache:
                _cache['logs'] = disk_cache['logs']
                _cache['enriched_data'] = disk_cache['enriched_data']
                _cache['last_modified'] = cache_mtime
                print("Using disk cache")
                return jsonify(disk_cache['enriched_data'])

    # Recompute
    print(f"Cache miss - recomputing enriched data...")
    logs = read_all_logs()
    enriched_data = enrich_logs(logs)

    # Update memory cache
    _cache['logs'] = logs
    _cache['enriched_data'] = enriched_data
    _cache['last_modified'] = latest_mtime

    # Save to disk
    save_cache_to_disk({
        'logs': logs,
        'enriched_data': enriched_data,
        'last_modified': latest_mtime
    })

    print(f"Enriched {len(logs)} logs, graph has {len(enriched_data['workflow_graph']['nodes'])} nodes and {len(enriched_data['workflow_graph']['edges'])} edges")

    return jsonify(enriched_data)


@app.route('/api/health')
def health():
    """Health check endpoint."""
    # Use cached data if available
    if _cache['enriched_data'] is not None:
        enriched_data = _cache['enriched_data']
        log_count = len(_cache['logs'])
    else:
        logs = read_all_logs()
        log_count = len(logs)
        enriched_data = {'workflow_graph': {'nodes': [], 'edges': []}}

    return jsonify({
        "status": "ok",
        "log_dir": str(LOG_DIR.absolute()),
        "log_count": log_count,
        "graph_nodes": len(enriched_data.get('workflow_graph', {}).get('nodes', [])),
        "graph_edges": len(enriched_data.get('workflow_graph', {}).get('edges', [])),
        "cache_valid": _cache['enriched_data'] is not None
    })


if __name__ == "__main__":
    print(f"Starting log API server on port {API_PORT}")
    print(f"Reading logs from: {LOG_DIR.absolute()}")
    app.run(host="127.0.0.1", port=API_PORT, debug=False)

