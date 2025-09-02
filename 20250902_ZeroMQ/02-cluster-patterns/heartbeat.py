#!/usr/bin/env python3
"""
Heartbeat and Failure Detection for ZeroMQ Cluster

This implements robust heartbeat mechanisms and failure detection:
- Bidirectional heartbeats between coordinator and workers
- Configurable timeout and retry logic
- Graceful handling of network partitions
- Automatic recovery and reconnection
"""

import zmq
import zmq.asyncio
import asyncio
import json
import time
import logging
import sys
from typing import Dict, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import uuid


class NodeState(Enum):
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    ALIVE = "alive"
    SUSPECTED = "suspected"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class HeartbeatConfig:
    interval: float = 5.0           # Heartbeat interval in seconds
    timeout: float = 15.0           # Timeout before marking as suspected
    failure_threshold: int = 3      # Missed heartbeats before marking as failed
    recovery_timeout: float = 30.0  # Time to wait for recovery
    max_retries: int = 5           # Max reconnection attempts


@dataclass
class NodeStatus:
    node_id: str
    state: NodeState
    last_heartbeat: float
    missed_heartbeats: int = 0
    retry_count: int = 0
    last_state_change: float = 0.0
    
    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp and reset missed count."""
        self.last_heartbeat = time.time()
        self.missed_heartbeats = 0
        if self.state in [NodeState.SUSPECTED, NodeState.FAILED]:
            self.state = NodeState.ALIVE
            self.last_state_change = time.time()
    
    def mark_missed_heartbeat(self) -> bool:
        """Mark a missed heartbeat. Returns True if state changed."""
        self.missed_heartbeats += 1
        old_state = self.state
        
        if self.missed_heartbeats >= 3 and self.state == NodeState.ALIVE:
            self.state = NodeState.SUSPECTED
            self.last_state_change = time.time()
        elif self.missed_heartbeats >= 5 and self.state == NodeState.SUSPECTED:
            self.state = NodeState.FAILED
            self.last_state_change = time.time()
        
        return old_state != self.state


class HeartbeatManager:
    """
    Manages heartbeats and failure detection for cluster nodes.
    """
    
    def __init__(self, 
                 node_id: str,
                 config: HeartbeatConfig,
                 is_coordinator: bool = False):
        """
        Initialize heartbeat manager.
        
        Args:
            node_id: Unique identifier for this node
            config: Heartbeat configuration
            is_coordinator: Whether this is the coordinator node
        """
        self.node_id = node_id
        self.config = config
        self.is_coordinator = is_coordinator
        
        # Node tracking (coordinator only)
        self.nodes: Dict[str, NodeStatus] = {}
        
        # Callbacks
        self.on_node_alive: Optional[Callable[[str], None]] = None
        self.on_node_suspected: Optional[Callable[[str], None]] = None
        self.on_node_failed: Optional[Callable[[str], None]] = None
        self.on_node_recovered: Optional[Callable[[str], None]] = None
        
        # ZeroMQ setup
        self.context = zmq.asyncio.Context()
        self.heartbeat_socket = None  # PUB for sending heartbeats
        self.monitor_socket = None    # SUB for receiving heartbeats
        
        # Control
        self.running = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"heartbeat-{node_id}")
    
    async def start(self, heartbeat_port: int, monitor_port: int) -> None:
        """
        Start the heartbeat manager.
        
        Args:
            heartbeat_port: Port for sending heartbeats
            monitor_port: Port for receiving heartbeats
        """
        self.logger.info(f"Starting heartbeat manager for {self.node_id}")
        
        # Setup sockets
        self.heartbeat_socket = self.context.socket(zmq.PUB)
        if self.is_coordinator:
            self.heartbeat_socket.bind(f"tcp://*:{heartbeat_port}")
        else:
            self.heartbeat_socket.connect(f"tcp://localhost:{heartbeat_port}")
        
        self.monitor_socket = self.context.socket(zmq.SUB)
        if self.is_coordinator:
            self.monitor_socket.bind(f"tcp://*:{monitor_port}")
        else:
            self.monitor_socket.connect(f"tcp://localhost:{monitor_port}")
        
        # Subscribe to all heartbeats
        self.monitor_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.running = True
        
        # Start background tasks
        await asyncio.gather(
            self._send_heartbeats(),
            self._monitor_heartbeats(),
            self._check_node_health() if self.is_coordinator else self._dummy_task()
        )
    
    async def stop(self) -> None:
        """Stop the heartbeat manager."""
        self.logger.info(f"Stopping heartbeat manager for {self.node_id}")
        self.running = False
        
        if self.heartbeat_socket:
            self.heartbeat_socket.close()
        if self.monitor_socket:
            self.monitor_socket.close()
        
        self.context.term()
    
    async def _dummy_task(self) -> None:
        """Dummy task for non-coordinator nodes."""
        while self.running:
            await asyncio.sleep(1)
    
    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeats."""
        while self.running:
            try:
                heartbeat = {
                    'type': 'heartbeat',
                    'node_id': self.node_id,
                    'timestamp': time.time(),
                    'is_coordinator': self.is_coordinator
                }
                
                await self.heartbeat_socket.send_string(json.dumps(heartbeat))
                self.logger.debug(f"Sent heartbeat from {self.node_id}")
                
                await asyncio.sleep(self.config.interval)
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
    
    async def _monitor_heartbeats(self) -> None:
        """Monitor incoming heartbeats."""
        while self.running:
            try:
                # Non-blocking receive with timeout
                try:
                    message_data = await asyncio.wait_for(
                        self.monitor_socket.recv_string(),
                        timeout=1.0
                    )
                    message = json.loads(message_data)
                    await self._process_heartbeat(message)
                    
                except asyncio.TimeoutError:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error monitoring heartbeats: {e}")
    
    async def _process_heartbeat(self, message: Dict) -> None:
        """Process a received heartbeat."""
        if message.get('type') != 'heartbeat':
            return
        
        sender_id = message.get('node_id')
        if not sender_id or sender_id == self.node_id:
            return  # Ignore our own heartbeats
        
        timestamp = message.get('timestamp', time.time())
        
        # Update or create node status
        if sender_id not in self.nodes:
            self.nodes[sender_id] = NodeStatus(
                node_id=sender_id,
                state=NodeState.ALIVE,
                last_heartbeat=timestamp,
                last_state_change=timestamp
            )
            self.logger.info(f"New node detected: {sender_id}")
            
            if self.on_node_alive:
                self.on_node_alive(sender_id)
        else:
            old_state = self.nodes[sender_id].state
            self.nodes[sender_id].update_heartbeat()
            
            # Check for recovery
            if old_state in [NodeState.SUSPECTED, NodeState.FAILED]:
                self.logger.info(f"Node {sender_id} recovered from {old_state.value}")
                if self.on_node_recovered:
                    self.on_node_recovered(sender_id)
        
        self.logger.debug(f"Received heartbeat from {sender_id}")
    
    async def _check_node_health(self) -> None:
        """Check health of all nodes (coordinator only)."""
        if not self.is_coordinator:
            return
        
        while self.running:
            try:
                current_time = time.time()
                
                for node_id, status in self.nodes.items():
                    time_since_heartbeat = current_time - status.last_heartbeat
                    
                    # Check for missed heartbeats
                    if time_since_heartbeat > self.config.interval * 2:
                        state_changed = status.mark_missed_heartbeat()
                        
                        if state_changed:
                            self.logger.warning(
                                f"Node {node_id} state changed to {status.state.value} "
                                f"(missed {status.missed_heartbeats} heartbeats)"
                            )
                            
                            # Trigger callbacks
                            if status.state == NodeState.SUSPECTED and self.on_node_suspected:
                                self.on_node_suspected(node_id)
                            elif status.state == NodeState.FAILED and self.on_node_failed:
                                self.on_node_failed(node_id)
                
                await asyncio.sleep(self.config.interval)
                
            except Exception as e:
                self.logger.error(f"Error checking node health: {e}")
    
    def register_callbacks(self,
                          on_alive: Optional[Callable[[str], None]] = None,
                          on_suspected: Optional[Callable[[str], None]] = None,
                          on_failed: Optional[Callable[[str], None]] = None,
                          on_recovered: Optional[Callable[[str], None]] = None) -> None:
        """
        Register callbacks for node state changes.
        
        Args:
            on_alive: Called when a new node is detected
            on_suspected: Called when a node is suspected of failure
            on_failed: Called when a node is marked as failed
            on_recovered: Called when a failed node recovers
        """
        self.on_node_alive = on_alive
        self.on_node_suspected = on_suspected
        self.on_node_failed = on_failed
        self.on_node_recovered = on_recovered
    
    def get_node_states(self) -> Dict[str, Dict]:
        """Get current state of all nodes."""
        return {
            node_id: {
                'state': status.state.value,
                'last_heartbeat': status.last_heartbeat,
                'missed_heartbeats': status.missed_heartbeats,
                'time_since_heartbeat': time.time() - status.last_heartbeat
            }
            for node_id, status in self.nodes.items()
        }
    
    def get_alive_nodes(self) -> Set[str]:
        """Get set of nodes that are currently alive."""
        return {
            node_id for node_id, status in self.nodes.items()
            if status.state == NodeState.ALIVE
        }
    
    def get_failed_nodes(self) -> Set[str]:
        """Get set of nodes that are currently failed."""
        return {
            node_id for node_id, status in self.nodes.items()
            if status.state == NodeState.FAILED
        }


class FailureDetector:
    """
    Advanced failure detector with phi-accrual algorithm.
    """
    
    def __init__(self, threshold: float = 8.0, max_sample_size: int = 1000):
        """
        Initialize failure detector.
        
        Args:
            threshold: Phi threshold for failure detection
            max_sample_size: Maximum number of samples to keep
        """
        self.threshold = threshold
        self.max_sample_size = max_sample_size
        self.arrival_intervals: Dict[str, List[float]] = {}
        self.last_arrival: Dict[str, float] = {}
    
    def heartbeat_arrived(self, node_id: str) -> None:
        """Record a heartbeat arrival."""
        current_time = time.time()
        
        if node_id in self.last_arrival:
            interval = current_time - self.last_arrival[node_id]
            
            if node_id not in self.arrival_intervals:
                self.arrival_intervals[node_id] = []
            
            self.arrival_intervals[node_id].append(interval)
            
            # Keep only recent samples
            if len(self.arrival_intervals[node_id]) > self.max_sample_size:
                self.arrival_intervals[node_id].pop(0)
        
        self.last_arrival[node_id] = current_time
    
    def phi(self, node_id: str) -> float:
        """Calculate phi value for a node."""
        if node_id not in self.last_arrival or node_id not in self.arrival_intervals:
            return 0.0
        
        intervals = self.arrival_intervals[node_id]
        if len(intervals) < 2:
            return 0.0
        
        # Calculate mean and variance
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        # Time since last heartbeat
        time_since_last = time.time() - self.last_arrival[node_id]
        
        # Calculate phi
        phi_value = time_since_last / mean_interval
        return max(0.0, phi_value)
    
    def is_available(self, node_id: str) -> bool:
        """Check if a node is available based on phi threshold."""
        return self.phi(node_id) < self.threshold


async def demo_heartbeat():
    """Demonstrate heartbeat system with coordinator and workers."""
    print("Starting heartbeat demo...")
    
    config = HeartbeatConfig(interval=2.0, timeout=6.0)
    
    # Start coordinator
    coordinator = HeartbeatManager("coordinator", config, is_coordinator=True)
    
    def on_node_alive(node_id: str):
        print(f"✓ Node {node_id} is alive")
    
    def on_node_suspected(node_id: str):
        print(f"⚠ Node {node_id} is suspected of failure")
    
    def on_node_failed(node_id: str):
        print(f"✗ Node {node_id} has failed")
    
    def on_node_recovered(node_id: str):
        print(f"↻ Node {node_id} has recovered")
    
    coordinator.register_callbacks(
        on_alive=on_node_alive,
        on_suspected=on_node_suspected,
        on_failed=on_node_failed,
        on_recovered=on_node_recovered
    )
    
    # Start workers
    workers = []
    for i in range(3):
        worker = HeartbeatManager(f"worker-{i}", config, is_coordinator=False)
        workers.append(worker)
    
    try:
        # Start all components
        tasks = [coordinator.start(5580, 5581)]
        for worker in workers:
            tasks.append(worker.start(5581, 5580))
        
        await asyncio.gather(*tasks)
        
    except KeyboardInterrupt:
        print("\nShutting down heartbeat demo...")
    finally:
        await coordinator.stop()
        for worker in workers:
            await worker.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_heartbeat())
    else:
        print("Usage: python heartbeat.py demo")
        print("This will start a heartbeat demo with coordinator and workers")
