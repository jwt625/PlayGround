#!/usr/bin/env python3
"""
Simple API server to serve log files to the viewer.
"""

import json
import os
import pickle
import logging
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from log_classifier import enrich_logs_only
from workflow_graph import build_workflow_graph

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

# Flag to prevent concurrent cache building
_cache_building = False


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
            logger.info(f"Loaded cache from disk: {len(cached.get('logs', []))} logs")
            return cached
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None


def save_cache_to_disk(cache_data):
    """Save enriched data cache to disk."""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Saved cache to disk: {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


@app.route('/api/logs')
def get_logs():
    """Return all logs with basic enrichment (agent type, tool info) but no workflow graph."""
    latest_mtime = get_latest_log_mtime()

    # Try memory cache first
    if (_cache['logs'] is not None and
        _cache['last_modified'] is not None and
        latest_mtime is not None and
        latest_mtime <= _cache['last_modified']):
        logger.info("Using memory cache for enriched logs")
        return jsonify({'logs': _cache['logs']})

    # Read logs from disk
    logger.info("Reading and enriching logs...")
    raw_logs = read_all_logs()
    logger.info(f"Read {len(raw_logs)} logs")

    # Enrich logs with metadata (agent type, tool info, etc.) - NO workflow graph
    enriched_logs = enrich_logs_only(raw_logs)
    logger.info(f"Enriched {len(enriched_logs)} logs with metadata")

    # Update memory cache
    _cache['logs'] = enriched_logs
    _cache['last_modified'] = latest_mtime
    logger.info("Memory cache updated with enriched logs")

    return jsonify({'logs': enriched_logs})


@app.route('/api/workflow')
def get_workflow():
    """Return workflow graph with enriched metadata (expensive, on-demand only)."""
    global _cache_building

    # If cache is being built, wait and return cached result
    if _cache_building:
        import time
        logger.info("Workflow graph is being built, waiting...")
        # Wait for cache to be ready (check every 0.5 seconds for up to 60 seconds)
        for _ in range(120):
            time.sleep(0.5)
            if not _cache_building:
                if _cache['enriched_data'] is not None:
                    logger.info("Workflow graph ready, returning cached data")
                    return jsonify(_cache['enriched_data'])
                else:
                    # Cache building failed, break and try again
                    break

        # If still building after timeout, return error
        if _cache_building:
            logger.error("Workflow graph building timeout")
            return jsonify({'error': 'Workflow graph building timeout'}), 503

    latest_mtime = get_latest_log_mtime()

    # Try memory cache first
    if (_cache['enriched_data'] is not None and
        _cache['last_modified'] is not None and
        latest_mtime is not None and
        latest_mtime <= _cache['last_modified']):
        logger.info("Using memory cache for workflow graph")
        return jsonify(_cache['enriched_data'])

    # Try disk cache
    if latest_mtime is not None and CACHE_FILE.exists():
        disk_cache = load_cache_from_disk()
        if disk_cache and disk_cache.get('last_modified'):
            # Check if disk cache is still valid
            if disk_cache['last_modified'] >= latest_mtime:
                _cache['logs'] = disk_cache['logs']
                _cache['enriched_data'] = disk_cache['enriched_data']
                _cache['last_modified'] = disk_cache['last_modified']
                logger.info("Using disk cache for workflow graph")
                return jsonify(disk_cache['enriched_data'])

    # Set flag to prevent concurrent builds
    _cache_building = True

    try:
        # Recompute
        logger.info("Workflow graph cache miss - building workflow graph...")

        # Use cached enriched logs if available and fresh
        if (_cache['logs'] is not None and
            _cache['last_modified'] is not None and
            latest_mtime is not None and
            latest_mtime <= _cache['last_modified']):
            enriched_logs = _cache['logs']
            logger.info(f"Using cached enriched logs: {len(enriched_logs)} logs")
        else:
            # Read and enrich logs
            raw_logs = read_all_logs()
            logger.info(f"Read {len(raw_logs)} logs from disk")
            enriched_logs = enrich_logs_only(raw_logs)
            logger.info(f"Enriched {len(enriched_logs)} logs with metadata")
            _cache['logs'] = enriched_logs
            _cache['last_modified'] = latest_mtime

        # Build workflow graph (expensive operation)
        logger.info("Building workflow graph...")
        workflow_graph = build_workflow_graph(enriched_logs)
        logger.info(f"Workflow graph built: {len(workflow_graph['nodes'])} nodes and {len(workflow_graph['edges'])} edges")

        enriched_data = {
            'logs': enriched_logs,
            'workflow_graph': workflow_graph
        }

        # Update memory cache
        _cache['enriched_data'] = enriched_data
        logger.info("Memory cache updated with workflow graph")

        # Save to disk
        save_cache_to_disk({
            'logs': enriched_logs,
            'enriched_data': enriched_data,
            'last_modified': latest_mtime
        })

        return jsonify(enriched_data)
    except Exception as e:
        logger.error(f"ERROR during workflow graph build: {e}", exc_info=True)
        # Clear cache on error so next request will retry
        _cache['enriched_data'] = None
        raise
    finally:
        _cache_building = False
        logger.info("Workflow graph building complete")


@app.route('/api/entities')
def get_entities():
    """Return entities JSON file."""
    # Find the most recent entities file
    entities_files = sorted(
        LOG_DIR.glob("entities_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not entities_files:
        return jsonify({"error": "No entities file found"}), 404

    try:
        with open(entities_files[0], 'r') as f:
            entities = json.load(f)
        return jsonify(entities)
    except Exception as e:
        logger.error(f"Failed to load entities: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint."""
    # Use cached data if available
    if _cache['logs'] is not None:
        log_count = len(_cache['logs'])
    else:
        logs = read_all_logs()
        log_count = len(logs)

    graph_nodes = 0
    graph_edges = 0
    if _cache['enriched_data'] is not None:
        graph_nodes = len(_cache['enriched_data'].get('workflow_graph', {}).get('nodes', []))
        graph_edges = len(_cache['enriched_data'].get('workflow_graph', {}).get('edges', []))

    return jsonify({
        "status": "ok",
        "log_dir": str(LOG_DIR.absolute()),
        "log_count": log_count,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "logs_cached": _cache['logs'] is not None,
        "workflow_cached": _cache['enriched_data'] is not None
    })


if __name__ == "__main__":
    logger.info(f"Starting log API server on port {API_PORT}")
    logger.info(f"Reading logs from: {LOG_DIR.absolute()}")
    logger.info(f"Cache file: {CACHE_FILE}")
    app.run(host="127.0.0.1", port=API_PORT, debug=False)

