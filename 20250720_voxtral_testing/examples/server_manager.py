#!/usr/bin/env python3
"""Server management script for Voxtral vLLM server."""

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.exceptions import VoxtralError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ServerManager:
    """Manages the vLLM server for Voxtral."""
    
    def __init__(self, config: VoxtralConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen] = None
    
    def start_server(self) -> bool:
        """Start the vLLM server.
        
        Returns:
            True if server started successfully, False otherwise.
        """
        if self.process and self.process.poll() is None:
            logger.warning("Server is already running")
            return True
        
        try:
            cmd = ["vllm", "serve"] + self.config.to_server_args()
            logger.info(f"Starting server with command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(5)
            
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"Server failed to start. Exit code: {self.process.returncode}")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
            
            logger.info(f"Server started with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop the vLLM server.
        
        Returns:
            True if server stopped successfully, False otherwise.
        """
        if not self.process or self.process.poll() is not None:
            logger.warning("Server is not running")
            return True
        
        try:
            logger.info(f"Stopping server with PID: {self.process.pid}")
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, forcing kill")
                self.process.kill()
                self.process.wait()
            
            logger.info("Server stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if the server process is running."""
        return self.process is not None and self.process.poll() is None
    
    async def wait_for_health(self, timeout: int = 60) -> bool:
        """Wait for server to become healthy.
        
        Args:
            timeout: Maximum time to wait in seconds.
            
        Returns:
            True if server becomes healthy, False if timeout.
        """
        client = VoxtralClient(self.config)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if await client.health_check():
                    logger.info("Server is healthy")
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        logger.error(f"Server did not become healthy within {timeout} seconds")
        return False


async def main() -> None:
    """Main function for server management."""
    parser = argparse.ArgumentParser(description="Manage Voxtral vLLM server")
    parser.add_argument(
        "action", 
        choices=["start", "stop", "restart", "status", "test"],
        help="Action to perform"
    )
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Health check timeout in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = VoxtralConfig(
        server_host=args.host,
        server_port=args.port
    )
    
    manager = ServerManager(config)
    
    if args.action == "start":
        if manager.start_server():
            if await manager.wait_for_health(args.timeout):
                logger.info("Server started and is healthy")
            else:
                logger.error("Server started but failed health check")
        else:
            logger.error("Failed to start server")
    
    elif args.action == "stop":
        if manager.stop_server():
            logger.info("Server stopped successfully")
        else:
            logger.error("Failed to stop server")
    
    elif args.action == "restart":
        logger.info("Restarting server...")
        manager.stop_server()
        time.sleep(2)
        if manager.start_server():
            if await manager.wait_for_health(args.timeout):
                logger.info("Server restarted and is healthy")
            else:
                logger.error("Server restarted but failed health check")
        else:
            logger.error("Failed to restart server")
    
    elif args.action == "status":
        if manager.is_running():
            client = VoxtralClient(config)
            if await client.health_check():
                logger.info("Server is running and healthy")
            else:
                logger.warning("Server is running but not healthy")
        else:
            logger.info("Server is not running")
    
    elif args.action == "test":
        client = VoxtralClient(config)
        try:
            if await client.health_check():
                model_name = await client.get_model_name()
                logger.info(f"Server test successful. Model: {model_name}")
            else:
                logger.error("Server test failed - health check failed")
        except Exception as e:
            logger.error(f"Server test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
