#!/usr/bin/env python3
"""
Startup script for the Voxtral FastAPI server.

This script provides a convenient way to start the FastAPI server with
proper configuration and environment setup.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
from app.core.config import AppConfig


def main() -> None:
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description="Start the Voxtral FastAPI server")
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to (default: 8080)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--voxtral-host",
        default="localhost",
        help="Voxtral backend host (default: localhost)"
    )
    
    parser.add_argument(
        "--voxtral-port",
        type=int,
        default=8000,
        help="Voxtral backend port (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables for configuration
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    os.environ["DEBUG"] = str(args.debug).lower()
    os.environ["VOXTRAL_HOST"] = args.voxtral_host
    os.environ["VOXTRAL_PORT"] = str(args.voxtral_port)
    
    # Create configuration
    config = AppConfig.from_env()
    
    print(f"Starting {config.app_name} v{config.app_version}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"Voxtral backend: http://{args.voxtral_host}:{args.voxtral_port}")
    print(f"Debug mode: {args.debug}")
    print(f"Auto-reload: {args.reload}")
    print()
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="debug" if args.debug else "info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
