#!/usr/bin/env python3
"""
Cluster Coordinator using ZeroMQ

This implements a central coordination node that:
- Manages cluster membership and health
- Distributes work to worker nodes
- Collects results and status updates
- Handles node failures and recovery
"""

import zmq
import zmq.asyncio
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum


class NodeStatus(Enum):
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class NodeInfo:
    node_id: str
    status: NodeStatus
    last_heartbeat: float
    capabilities: Dict[str, any]
    current_task: Optional[str] = None
    error_count: int = 0


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: Dict[str, any]
    assigned_node: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, any]] = None
    error: Optional[str] = None


class ClusterCoordinator:
    """
    Central coordinator for managing a cluster of worker nodes.
    """
    
    def __init__(self, 
                 worker_port: int = 5570,
                 status_port: int = 5571,
                 heartbeat_interval: float = 5.0,
                 heartbeat_timeout: float = 15.0):
        """
        Initialize the cluster coordinator.
        
        Args:
            worker_port: Port for worker task distribution (ROUTER)
            status_port: Port for status updates and heartbeats (SUB)
            heartbeat_interval: How often to expect heartbeats (seconds)
            heartbeat_timeout: When to consider a node dead (seconds)
        """
        self.worker_port = worker_port
        self.status_port = status_port
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # ZeroMQ setup
        self.context = zmq.asyncio.Context()
        self.worker_socket = None  # ROUTER for task distribution
        self.status_socket = None  # SUB for status updates
        
        # Cluster state
        self.nodes: Dict[str, NodeInfo] = {}
        self.pending_tasks: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        
        # Control
        self.running = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the coordinator."""
        self.logger.info("Starting cluster coordinator...")
        
        # Setup sockets
        self.worker_socket = self.context.socket(zmq.ROUTER)
        self.worker_socket.bind(f"tcp://*:{self.worker_port}")
        
        self.status_socket = self.context.socket(zmq.SUB)
        self.status_socket.bind(f"tcp://*:{self.status_port}")
        self.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
        
        self.running = True
        
        # Start background tasks
        await asyncio.gather(
            self._handle_worker_messages(),
            self._handle_status_updates(),
            self._monitor_heartbeats(),
            self._schedule_tasks()
        )
    
    async def stop(self) -> None:
        """Stop the coordinator."""
        self.logger.info("Stopping cluster coordinator...")
        self.running = False
        
        if self.worker_socket:
            self.worker_socket.close()
        if self.status_socket:
            self.status_socket.close()
        
        self.context.term()
    
    async def _handle_worker_messages(self) -> None:
        """Handle messages from worker nodes."""
        while self.running:
            try:
                # Receive message from worker
                worker_id, empty, message_data = await self.worker_socket.recv_multipart()
                worker_id = worker_id.decode('utf-8')
                message = json.loads(message_data.decode('utf-8'))
                
                await self._process_worker_message(worker_id, message)
                
            except Exception as e:
                self.logger.error(f"Error handling worker message: {e}")
    
    async def _handle_status_updates(self) -> None:
        """Handle status updates and heartbeats."""
        while self.running:
            try:
                message_data = await self.status_socket.recv_string()
                message = json.loads(message_data)
                
                await self._process_status_update(message)
                
            except Exception as e:
                self.logger.error(f"Error handling status update: {e}")
    
    async def _process_worker_message(self, worker_id: str, message: Dict) -> None:
        """Process a message from a worker node."""
        msg_type = message.get('type')
        
        if msg_type == 'register':
            await self._register_node(worker_id, message.get('capabilities', {}))
        elif msg_type == 'task_result':
            await self._handle_task_result(worker_id, message)
        elif msg_type == 'error':
            await self._handle_worker_error(worker_id, message)
        else:
            self.logger.warning(f"Unknown message type from {worker_id}: {msg_type}")
    
    async def _process_status_update(self, message: Dict) -> None:
        """Process a status update message."""
        msg_type = message.get('type')
        node_id = message.get('node_id')
        
        if msg_type == 'heartbeat':
            await self._update_heartbeat(node_id, message.get('status'))
        elif msg_type == 'status_change':
            await self._update_node_status(node_id, message.get('status'))
    
    async def _register_node(self, node_id: str, capabilities: Dict) -> None:
        """Register a new worker node."""
        self.logger.info(f"Registering node: {node_id}")
        
        node_info = NodeInfo(
            node_id=node_id,
            status=NodeStatus.READY,
            last_heartbeat=time.time(),
            capabilities=capabilities
        )
        
        self.nodes[node_id] = node_info
        
        # Send acknowledgment
        response = {
            'type': 'register_ack',
            'coordinator_id': 'coordinator',
            'heartbeat_interval': self.heartbeat_interval
        }
        
        await self.worker_socket.send_multipart([
            node_id.encode('utf-8'),
            b'',
            json.dumps(response).encode('utf-8')
        ])
    
    async def _handle_task_result(self, worker_id: str, message: Dict) -> None:
        """Handle task completion from worker."""
        task_id = message.get('task_id')
        result = message.get('result')
        error = message.get('error')
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.completed_at = time.time()
            task.result = result
            task.error = error
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            # Update node status
            if worker_id in self.nodes:
                self.nodes[worker_id].status = NodeStatus.READY
                self.nodes[worker_id].current_task = None
            
            if error:
                self.logger.error(f"Task {task_id} failed on {worker_id}: {error}")
            else:
                self.logger.info(f"Task {task_id} completed on {worker_id}")
    
    async def _handle_worker_error(self, worker_id: str, message: Dict) -> None:
        """Handle error from worker."""
        error_msg = message.get('error', 'Unknown error')
        self.logger.error(f"Worker {worker_id} error: {error_msg}")
        
        if worker_id in self.nodes:
            self.nodes[worker_id].error_count += 1
            self.nodes[worker_id].status = NodeStatus.ERROR
    
    async def _update_heartbeat(self, node_id: str, status: str) -> None:
        """Update node heartbeat."""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
            if status:
                self.nodes[node_id].status = NodeStatus(status)
    
    async def _update_node_status(self, node_id: str, status: str) -> None:
        """Update node status."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus(status)
            self.logger.info(f"Node {node_id} status: {status}")
    
    async def _monitor_heartbeats(self) -> None:
        """Monitor node heartbeats and detect failures."""
        while self.running:
            current_time = time.time()
            
            for node_id, node_info in self.nodes.items():
                time_since_heartbeat = current_time - node_info.last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    if node_info.status != NodeStatus.DISCONNECTED:
                        self.logger.warning(f"Node {node_id} heartbeat timeout")
                        node_info.status = NodeStatus.DISCONNECTED
                        
                        # Handle any active task on this node
                        if node_info.current_task:
                            await self._reschedule_task(node_info.current_task)
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _schedule_tasks(self) -> None:
        """Schedule pending tasks to available workers."""
        while self.running:
            if self.pending_tasks:
                available_nodes = [
                    node_id for node_id, node_info in self.nodes.items()
                    if node_info.status == NodeStatus.READY
                ]
                
                while self.pending_tasks and available_nodes:
                    task = self.pending_tasks.pop(0)
                    node_id = available_nodes.pop(0)
                    
                    await self._assign_task(task, node_id)
            
            await asyncio.sleep(1.0)
    
    async def _assign_task(self, task: Task, node_id: str) -> None:
        """Assign a task to a worker node."""
        task.assigned_node = node_id
        task.started_at = time.time()
        
        # Update node status
        self.nodes[node_id].status = NodeStatus.BUSY
        self.nodes[node_id].current_task = task.task_id
        
        # Move to active tasks
        self.active_tasks[task.task_id] = task
        
        # Send task to worker
        task_message = {
            'type': 'task',
            'task_id': task.task_id,
            'task_type': task.task_type,
            'payload': task.payload
        }
        
        await self.worker_socket.send_multipart([
            node_id.encode('utf-8'),
            b'',
            json.dumps(task_message).encode('utf-8')
        ])
        
        self.logger.info(f"Assigned task {task.task_id} to node {node_id}")
    
    async def _reschedule_task(self, task_id: str) -> None:
        """Reschedule a task due to node failure."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            del self.active_tasks[task_id]
            
            # Reset task state
            task.assigned_node = None
            task.started_at = None
            
            # Add back to pending
            self.pending_tasks.insert(0, task)
            
            self.logger.info(f"Rescheduled task {task_id} due to node failure")
    
    async def submit_task(self, task_type: str, payload: Dict[str, any]) -> str:
        """Submit a new task to the cluster."""
        task_id = f"task_{int(time.time() * 1000)}"
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            created_at=time.time()
        )
        
        self.pending_tasks.append(task)
        self.logger.info(f"Submitted task {task_id} of type {task_type}")
        
        return task_id
    
    def get_cluster_status(self) -> Dict:
        """Get current cluster status."""
        return {
            'nodes': {
                node_id: {
                    'status': node_info.status.value,
                    'last_heartbeat': node_info.last_heartbeat,
                    'current_task': node_info.current_task,
                    'error_count': node_info.error_count
                }
                for node_id, node_info in self.nodes.items()
            },
            'tasks': {
                'pending': len(self.pending_tasks),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks)
            }
        }


async def main():
    """Run the coordinator."""
    coordinator = ClusterCoordinator()
    
    try:
        # Start coordinator
        await coordinator.start()
    except KeyboardInterrupt:
        print("\nShutting down coordinator...")
    finally:
        await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(main())
