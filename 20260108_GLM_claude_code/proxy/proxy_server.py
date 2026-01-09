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
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, request
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


def parse_sse_stream(stream_data: str) -> dict[str, Any]:
    """
    Parse Server-Sent Events stream and aggregate into a single response.

    Returns a dict with the aggregated message content and metadata.
    """
    lines = stream_data.strip().split('\n')

    message_data = {
        "id": None,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": None,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": None
    }

    current_content_block = None
    current_event = None
    current_data = ""

    for line in lines:
        line = line.strip()

        if line.startswith('event: '):
            current_event = line[7:]
            current_data = ""
        elif line.startswith('data: '):
            current_data = line[6:]

            try:
                data = json.loads(current_data)

                if current_event == 'message_start':
                    msg = data.get('message', {})
                    message_data['id'] = msg.get('id')
                    message_data['model'] = msg.get('model')
                    message_data['role'] = msg.get('role', 'assistant')
                    message_data['usage'] = msg.get('usage')

                elif current_event == 'content_block_start':
                    block = data.get('content_block', {})
                    current_content_block = {
                        'type': block.get('type'),
                        'text': '' if block.get('type') == 'text' else None
                    }
                    if block.get('type') == 'thinking':
                        current_content_block['thinking'] = ''

                elif current_event == 'content_block_delta':
                    delta = data.get('delta', {})
                    if current_content_block:
                        if delta.get('type') == 'text_delta':
                            current_content_block['text'] += delta.get('text', '')
                        elif delta.get('type') == 'thinking_delta':
                            current_content_block['thinking'] += delta.get('thinking', '')

                elif current_event == 'content_block_stop':
                    if current_content_block:
                        message_data['content'].append(current_content_block)
                        current_content_block = None

                elif current_event == 'message_delta':
                    delta = data.get('delta', {})
                    if 'stop_reason' in delta:
                        message_data['stop_reason'] = delta['stop_reason']
                    if 'stop_sequence' in delta:
                        message_data['stop_sequence'] = delta['stop_sequence']
                    usage = data.get('usage', {})
                    if usage:
                        if message_data['usage'] is None:
                            message_data['usage'] = {}
                        message_data['usage'].update(usage)

                elif current_event == 'message_stop':
                    pass

            except json.JSONDecodeError:
                continue

    return message_data


def is_sse_stream(content: bytes, headers: dict) -> bool:
    """Check if the response is a Server-Sent Events stream."""
    content_type = headers.get('content-type', headers.get('Content-Type', ''))
    if 'text/event-stream' in content_type:
        return True

    # Also check if content starts with SSE format
    try:
        text = content.decode('utf-8', errors='ignore')[:100]
        return text.startswith('event: ') or '\nevent: ' in text
    except:
        return False


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
    request_timestamp = datetime.utcnow()
    log_data = {
        "timestamp": request_timestamp.isoformat() + "Z",
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
        response_timestamp = datetime.utcnow()

        # Parse response body for logging
        response_body_parsed = None
        if response_body:
            # Check if this is an SSE stream
            if is_sse_stream(response_body, dict(upstream_response.headers)):
                try:
                    stream_text = response_body.decode("utf-8", errors="replace")
                    response_body_parsed = parse_sse_stream(stream_text)
                except Exception as e:
                    # If parsing fails, log the raw stream (truncated)
                    response_body_parsed = {
                        "error": f"Failed to parse SSE stream: {str(e)}",
                        "raw_preview": response_body.decode("utf-8", errors="replace")[:1000]
                    }
            else:
                try:
                    response_body_parsed = json.loads(response_body)
                except json.JSONDecodeError:
                    response_body_parsed = response_body.decode("utf-8", errors="replace")

        # Log response
        duration_ms = (response_timestamp - request_timestamp).total_seconds() * 1000
        log_data["response"] = {
            "status": upstream_response.status_code,
            "headers": redact_headers(dict(upstream_response.headers)),
            "body": response_body_parsed,
            "timestamp": response_timestamp.isoformat() + "Z",
            "duration_ms": round(duration_ms, 2),
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

