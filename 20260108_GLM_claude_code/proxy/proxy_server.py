#!/usr/bin/env python3
"""
Logging proxy server for Claude Code inference requests.
Forwards requests to the actual endpoint while logging all communication.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Request, Response, request
import requests

load_dotenv()

app = Flask(__name__)

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "").rstrip("/")
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
PROXY_PORT = int(os.getenv("PROXY_PORT", "58734"))
REDACT_KEYS = {"authorization", "api-key", "x-api-key", "cookie"}

LOG_DIR.mkdir(exist_ok=True)


def redact_headers(headers: dict) -> dict:
    """Remove sensitive headers from logging."""
    return {
        k: ("REDACTED" if k.lower() in REDACT_KEYS else v)
        for k, v in headers.items()
    }


def get_log_filename() -> Path:
    """Generate log filename with date."""
    return LOG_DIR / f"requests_{datetime.now().strftime('%Y%m%d')}.jsonl"


def log_entry(entry: dict) -> None:
    """Append log entry to JSONL file."""
    with open(get_log_filename(), "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy(path: str) -> Response:
    """Proxy all requests to upstream server with logging."""
    
    if not UPSTREAM_URL:
        return Response(
            json.dumps({"error": "UPSTREAM_URL not configured"}),
            status=500,
            content_type="application/json"
        )
    
    upstream_url = f"{UPSTREAM_URL}/{path}"
    
    # Prepare request data
    headers = {k: v for k, v in request.headers if k.lower() != "host"}
    body = request.get_data()
    
    # Parse request body for logging
    request_body = None
    if body:
        try:
            request_body = json.loads(body)
        except json.JSONDecodeError:
            request_body = body.decode("utf-8", errors="replace")
    
    # Log request
    log_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "direction": "request",
        "method": request.method,
        "path": path,
        "url": upstream_url,
        "headers": redact_headers(dict(request.headers)),
        "body": request_body,
    }
    
    try:
        # Forward request to upstream
        upstream_response = requests.request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            data=body,
            stream=True,
            timeout=600,
        )
        
        # Collect response data
        response_body = upstream_response.content
        
        # Parse response body for logging
        response_body_parsed = None
        if response_body:
            try:
                response_body_parsed = json.loads(response_body)
            except json.JSONDecodeError:
                response_body_parsed = response_body.decode("utf-8", errors="replace")
        
        # Log response
        log_data["response"] = {
            "status": upstream_response.status_code,
            "headers": redact_headers(dict(upstream_response.headers)),
            "body": response_body_parsed,
        }
        log_entry(log_data)
        
        # Return response to client
        return Response(
            response_body,
            status=upstream_response.status_code,
            headers=dict(upstream_response.headers),
        )
        
    except requests.exceptions.RequestException as e:
        # Log error
        log_data["error"] = str(e)
        log_entry(log_data)
        
        return Response(
            json.dumps({"error": f"Proxy error: {str(e)}"}),
            status=502,
            content_type="application/json"
        )


@app.route("/")
def health() -> Response:
    """Health check endpoint."""
    return Response(
        json.dumps({
            "status": "ok",
            "upstream": UPSTREAM_URL,
            "log_dir": str(LOG_DIR.absolute()),
        }),
        content_type="application/json"
    )


if __name__ == "__main__":
    if not UPSTREAM_URL:
        print("ERROR: UPSTREAM_URL environment variable is required", file=sys.stderr)
        sys.exit(1)
    
    print(f"Starting proxy server on port {PROXY_PORT}")
    print(f"Forwarding to: {UPSTREAM_URL}")
    print(f"Logging to: {LOG_DIR.absolute()}")
    
    app.run(host="127.0.0.1", port=PROXY_PORT, debug=False)

