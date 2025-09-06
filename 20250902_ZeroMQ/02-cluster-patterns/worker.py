#!/usr/bin/env python3
"""
Cluster Worker Node using ZeroMQ

This implements a worker node that:
- Registers with the cluster coordinator
- Receives and processes tasks
- Sends heartbeats and status updates
- Handles graceful shutdown and error recovery
"""

import zmq
import zmq.asyncio
import asyncio
import json
import time
import logging
import sys
import uuid
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class WorkerStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class WorkerConfig:
    node_id: str
    coordinator_host: str = "localhost"
    coordinator_worker_port: int = 5570
    coordinator_status_port: int = 5571
    heartbeat_interval: float = 5.0
    capabilities: Dict[str, Any] = None


class ClusterWorker:
    """
    Worker node that connects to cluster coordinator and processes tasks.
    """
    
    def __init__(self, config: WorkerConfig):
        """
        Initialize the cluster worker.
        
        Args:
            config: Worker configuration
        """
        self.config = config
        self.status = WorkerStatus.STARTING
        
        # ZeroMQ setup
        self.context = zmq.asyncio.Context()
        self.coordinator_socket = None  # DEALER for coordinator communication
        self.status_socket = None       # PUB for status updates
        
        # Task processing
        self.task_handlers: Dict[str, Callable] = {}
        self.current_task: Optional[str] = None
        
        # Control
        self.running = False
        self.heartbeat_interval = config.heartbeat_interval
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"worker-{config.node_id}")
        
        # Register default task handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default task handlers."""
        self.register_task_handler("echo", self._handle_echo_task)
        self.register_task_handler("compute", self._handle_compute_task)
        self.register_task_handler("sleep", self._handle_sleep_task)
    
    async def start(self) -> None:
        """Start the worker node."""
        self.logger.info(f"Starting worker node {self.config.node_id}")
        
        # Setup sockets
        self.coordinator_socket = self.context.socket(zmq.DEALER)
        self.coordinator_socket.setsockopt_string(zmq.IDENTITY, self.config.node_id)
        self.coordinator_socket.connect(
            f"tcp://{self.config.coordinator_host}:{self.config.coordinator_worker_port}"
        )
        
        self.status_socket = self.context.socket(zmq.PUB)
        self.status_socket.connect(
            f"tcp://{self.config.coordinator_host}:{self.config.coordinator_status_port}"
        )
        
        self.running = True
        
        # Register with coordinator
        await self._register_with_coordinator()
        
        # Start background tasks
        await asyncio.gather(
            self._handle_coordinator_messages(),
            self._send_heartbeats(),
            return_exceptions=True
        )
    
    async def stop(self) -> None:
        """Stop the worker node."""
        self.logger.info(f"Stopping worker node {self.config.node_id}")
        self.status = WorkerStatus.SHUTTING_DOWN
        self.running = False
        
        # Send final status update
        await self._send_status_update("shutting_down")
        
        if self.coordinator_socket:
            self.coordinator_socket.close()
        if self.status_socket:
            self.status_socket.close()
        
        self.context.term()
    
    async def _register_with_coordinator(self) -> None:
        """Register this worker with the coordinator."""
        capabilities = self.config.capabilities or {
            "task_types": list(self.task_handlers.keys()),
            "max_concurrent_tasks": 1
        }
        
        registration_message = {
            'type': 'register',
            'node_id': self.config.node_id,
            'capabilities': capabilities
        }
        
        await self.coordinator_socket.send_string(json.dumps(registration_message))
        self.logger.info("Sent registration to coordinator")
    
    async def _handle_coordinator_messages(self) -> None:
        """Handle messages from the coordinator."""
        while self.running:
            try:
                message_data = await self.coordinator_socket.recv_string()
                message = json.loads(message_data)
                
                await self._process_coordinator_message(message)
                
            except Exception as e:
                self.logger.error(f"Error handling coordinator message: {e}")
                await self._send_error_message(str(e))
    
    async def _process_coordinator_message(self, message: Dict) -> None:
        """Process a message from the coordinator."""
        msg_type = message.get('type')
        
        if msg_type == 'register_ack':
            await self._handle_registration_ack(message)
        elif msg_type == 'task':
            await self._handle_task_assignment(message)
        elif msg_type == 'ping':
            await self._handle_ping(message)
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
    
    async def _handle_registration_ack(self, message: Dict) -> None:
        """Handle registration acknowledgment from coordinator."""
        self.logger.info("Registration acknowledged by coordinator")
        self.status = WorkerStatus.READY
        
        # Update heartbeat interval if provided
        if 'heartbeat_interval' in message:
            self.heartbeat_interval = message['heartbeat_interval']
        
        await self._send_status_update("ready")
    
    async def _handle_task_assignment(self, message: Dict) -> None:
        """Handle task assignment from coordinator."""
        task_id = message.get('task_id')
        task_type = message.get('task_type')
        payload = message.get('payload', {})
        
        self.logger.info(f"Received task {task_id} of type {task_type}")
        
        if task_type not in self.task_handlers:
            error_msg = f"Unknown task type: {task_type}"
            await self._send_task_result(task_id, None, error_msg)
            return
        
        self.current_task = task_id
        self.status = WorkerStatus.BUSY
        await self._send_status_update("busy")
        
        try:
            # Execute task
            handler = self.task_handlers[task_type]
            result = await handler(payload)
            
            # Send result
            await self._send_task_result(task_id, result, None)
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            self.logger.error(error_msg)
            await self._send_task_result(task_id, None, error_msg)
        
        finally:
            self.current_task = None
            self.status = WorkerStatus.READY
            await self._send_status_update("ready")
    
    async def _handle_ping(self, message: Dict) -> None:
        """Handle ping from coordinator."""
        pong_message = {
            'type': 'pong',
            'node_id': self.config.node_id,
            'status': self.status.value
        }
        await self.coordinator_socket.send_string(json.dumps(pong_message))
    
    async def _send_task_result(self, task_id: str, result: Any, error: Optional[str]) -> None:
        """Send task result to coordinator."""
        result_message = {
            'type': 'task_result',
            'task_id': task_id,
            'node_id': self.config.node_id,
            'result': result,
            'error': error
        }
        
        await self.coordinator_socket.send_string(json.dumps(result_message))
        
        if error:
            self.logger.error(f"Task {task_id} failed: {error}")
        else:
            self.logger.info(f"Task {task_id} completed successfully")
    
    async def _send_error_message(self, error: str) -> None:
        """Send error message to coordinator."""
        error_message = {
            'type': 'error',
            'node_id': self.config.node_id,
            'error': error
        }
        
        await self.coordinator_socket.send_string(json.dumps(error_message))
    
    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to coordinator."""
        while self.running:
            await self._send_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self) -> None:
        """Send heartbeat to coordinator."""
        heartbeat_message = {
            'type': 'heartbeat',
            'node_id': self.config.node_id,
            'status': self.status.value,
            'timestamp': time.time(),
            'current_task': self.current_task
        }
        
        await self.status_socket.send_string(json.dumps(heartbeat_message))
    
    async def _send_status_update(self, status: str) -> None:
        """Send status update to coordinator."""
        status_message = {
            'type': 'status_change',
            'node_id': self.config.node_id,
            'status': status,
            'timestamp': time.time()
        }
        
        await self.status_socket.send_string(json.dumps(status_message))
    
    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """
        Register a task handler for a specific task type.
        
        Args:
            task_type: Type of task this handler processes
            handler: Async function that takes payload and returns result
        """
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    # Default task handlers
    
    async def _handle_echo_task(self, payload: Dict) -> Dict:
        """Handle echo task - simply returns the payload."""
        return {"echo": payload}
    
    async def _handle_compute_task(self, payload: Dict) -> Dict:
        """Handle compute task - performs some computation."""
        operation = payload.get('operation', 'add')
        numbers = payload.get('numbers', [1, 2, 3])
        
        if operation == 'add':
            result = sum(numbers)
        elif operation == 'multiply':
            result = 1
            for num in numbers:
                result *= num
        elif operation == 'max':
            result = max(numbers)
        elif operation == 'min':
            result = min(numbers)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Simulate some work
        await asyncio.sleep(0.5)
        
        return {
            "operation": operation,
            "input": numbers,
            "result": result
        }
    
    async def _handle_sleep_task(self, payload: Dict) -> Dict:
        """Handle sleep task - sleeps for specified duration."""
        duration = payload.get('duration', 1.0)
        
        self.logger.info(f"Sleeping for {duration} seconds")
        await asyncio.sleep(duration)
        
        return {
            "slept_for": duration,
            "completed_at": time.time()
        }


async def main():
    """Run a worker node."""
    if len(sys.argv) < 2:
        node_id = f"worker-{uuid.uuid4().hex[:8]}"
    else:
        node_id = sys.argv[1]
    
    config = WorkerConfig(
        node_id=node_id,
        coordinator_host="localhost",
        capabilities={
            "task_types": ["echo", "compute", "sleep"],
            "max_concurrent_tasks": 1
        }
    )
    
    worker = ClusterWorker(config)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        print(f"\nShutting down worker {node_id}...")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
