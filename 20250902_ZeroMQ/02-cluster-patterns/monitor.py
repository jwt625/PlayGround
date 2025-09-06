#!/usr/bin/env python3
"""
Cluster Monitor using ZeroMQ

This implements a monitoring system that:
- Observes cluster health and performance
- Collects metrics from coordinator and workers
- Provides real-time status dashboard
- Alerts on failures and performance issues
"""

import zmq
import zmq.asyncio
import asyncio
import json
import time
import logging
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics


@dataclass
class NodeMetrics:
    node_id: str
    status: str
    last_seen: float
    heartbeat_count: int = 0
    task_count: int = 0
    error_count: int = 0
    response_times: List[float] = field(default_factory=list)
    
    def add_response_time(self, response_time: float) -> None:
        """Add a response time measurement."""
        self.response_times.append(response_time)
        # Keep only last 100 measurements
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def get_avg_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    def get_health_score(self) -> float:
        """Calculate health score (0-100)."""
        base_score = 100.0
        
        # Deduct for errors
        if self.task_count > 0:
            error_rate = self.error_count / self.task_count
            base_score -= error_rate * 50
        
        # Deduct for slow response times
        avg_response = self.get_avg_response_time()
        if avg_response > 5.0:  # More than 5 seconds is concerning
            base_score -= min(30, (avg_response - 5.0) * 5)
        
        # Deduct for being offline
        time_since_seen = time.time() - self.last_seen
        if time_since_seen > 30:  # Haven't seen in 30 seconds
            base_score -= 40
        
        return max(0.0, base_score)


@dataclass
class ClusterMetrics:
    total_nodes: int = 0
    active_nodes: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_time: float = 0.0
    cluster_health: float = 0.0


class ClusterMonitor:
    """
    Monitor for observing cluster health and performance.
    """
    
    def __init__(self,
                 coordinator_host: str = "localhost",
                 status_port: int = 5571,
                 monitor_port: int = 5572,
                 update_interval: float = 5.0):
        """
        Initialize the cluster monitor.
        
        Args:
            coordinator_host: Coordinator hostname
            status_port: Port for receiving status updates
            monitor_port: Port for serving monitoring data
            update_interval: How often to update metrics (seconds)
        """
        self.coordinator_host = coordinator_host
        self.status_port = status_port
        self.monitor_port = monitor_port
        self.update_interval = update_interval
        
        # ZeroMQ setup
        self.context = zmq.asyncio.Context()
        self.status_socket = None    # SUB for status updates
        self.monitor_socket = None   # REP for serving monitoring data
        
        # Metrics storage
        self.node_metrics: Dict[str, NodeMetrics] = {}
        self.cluster_metrics = ClusterMetrics()
        self.task_history: List[Dict] = []
        
        # Control
        self.running = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the monitor."""
        self.logger.info("Starting cluster monitor...")
        
        # Setup sockets
        self.status_socket = self.context.socket(zmq.SUB)
        self.status_socket.connect(f"tcp://{self.coordinator_host}:{self.status_port}")
        self.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all
        
        self.monitor_socket = self.context.socket(zmq.REP)
        self.monitor_socket.bind(f"tcp://*:{self.monitor_port}")
        
        self.running = True
        
        # Start background tasks
        await asyncio.gather(
            self._collect_status_updates(),
            self._serve_monitoring_requests(),
            self._update_metrics(),
            self._log_periodic_status()
        )
    
    async def stop(self) -> None:
        """Stop the monitor."""
        self.logger.info("Stopping cluster monitor...")
        self.running = False
        
        if self.status_socket:
            self.status_socket.close()
        if self.monitor_socket:
            self.monitor_socket.close()
        
        self.context.term()
    
    async def _collect_status_updates(self) -> None:
        """Collect status updates from cluster nodes."""
        while self.running:
            try:
                message_data = await self.status_socket.recv_string()
                message = json.loads(message_data)
                
                await self._process_status_update(message)
                
            except Exception as e:
                self.logger.error(f"Error collecting status update: {e}")
    
    async def _serve_monitoring_requests(self) -> None:
        """Serve monitoring data requests."""
        while self.running:
            try:
                request = await self.monitor_socket.recv_string()
                response = await self._handle_monitoring_request(request)
                await self.monitor_socket.send_string(response)
                
            except Exception as e:
                self.logger.error(f"Error serving monitoring request: {e}")
    
    async def _process_status_update(self, message: Dict) -> None:
        """Process a status update message."""
        msg_type = message.get('type')
        node_id = message.get('node_id')
        
        if not node_id:
            return
        
        # Ensure node metrics exist
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = NodeMetrics(
                node_id=node_id,
                status="unknown",
                last_seen=time.time()
            )
        
        node_metrics = self.node_metrics[node_id]
        node_metrics.last_seen = time.time()
        
        if msg_type == 'heartbeat':
            node_metrics.heartbeat_count += 1
            node_metrics.status = message.get('status', 'unknown')
            
        elif msg_type == 'status_change':
            node_metrics.status = message.get('status', 'unknown')
            
        elif msg_type == 'task_completed':
            node_metrics.task_count += 1
            
            # Track task completion time
            task_duration = message.get('duration')
            if task_duration:
                node_metrics.add_response_time(task_duration)
            
            # Add to task history
            self.task_history.append({
                'node_id': node_id,
                'task_id': message.get('task_id'),
                'duration': task_duration,
                'timestamp': time.time(),
                'success': message.get('success', True)
            })
            
            # Keep only last 1000 tasks
            if len(self.task_history) > 1000:
                self.task_history.pop(0)
            
        elif msg_type == 'task_failed':
            node_metrics.error_count += 1
            
            # Add to task history
            self.task_history.append({
                'node_id': node_id,
                'task_id': message.get('task_id'),
                'error': message.get('error'),
                'timestamp': time.time(),
                'success': False
            })
    
    async def _handle_monitoring_request(self, request: str) -> str:
        """Handle a monitoring data request."""
        try:
            req_data = json.loads(request)
            req_type = req_data.get('type', 'status')
            
            if req_type == 'status':
                return json.dumps(self._get_cluster_status())
            elif req_type == 'nodes':
                return json.dumps(self._get_node_details())
            elif req_type == 'metrics':
                return json.dumps(self._get_cluster_metrics())
            elif req_type == 'history':
                limit = req_data.get('limit', 100)
                return json.dumps(self._get_task_history(limit))
            else:
                return json.dumps({'error': f'Unknown request type: {req_type}'})
                
        except Exception as e:
            return json.dumps({'error': str(e)})
    
    def _get_cluster_status(self) -> Dict:
        """Get overall cluster status."""
        current_time = time.time()
        active_nodes = 0
        healthy_nodes = 0
        
        for node_metrics in self.node_metrics.values():
            if current_time - node_metrics.last_seen < 30:  # Seen in last 30 seconds
                active_nodes += 1
                if node_metrics.get_health_score() > 70:
                    healthy_nodes += 1
        
        return {
            'timestamp': current_time,
            'total_nodes': len(self.node_metrics),
            'active_nodes': active_nodes,
            'healthy_nodes': healthy_nodes,
            'cluster_health': (healthy_nodes / max(1, active_nodes)) * 100,
            'total_tasks': len(self.task_history),
            'successful_tasks': len([t for t in self.task_history if t.get('success', True)]),
            'failed_tasks': len([t for t in self.task_history if not t.get('success', True)])
        }
    
    def _get_node_details(self) -> Dict:
        """Get detailed node information."""
        nodes = {}
        
        for node_id, metrics in self.node_metrics.items():
            nodes[node_id] = {
                'status': metrics.status,
                'last_seen': metrics.last_seen,
                'heartbeat_count': metrics.heartbeat_count,
                'task_count': metrics.task_count,
                'error_count': metrics.error_count,
                'avg_response_time': metrics.get_avg_response_time(),
                'health_score': metrics.get_health_score(),
                'online': time.time() - metrics.last_seen < 30
            }
        
        return {'nodes': nodes}
    
    def _get_cluster_metrics(self) -> Dict:
        """Get cluster-wide metrics."""
        if not self.task_history:
            return {'metrics': self.cluster_metrics.__dict__}
        
        # Calculate metrics from task history
        recent_tasks = [
            t for t in self.task_history
            if time.time() - t['timestamp'] < 3600  # Last hour
        ]
        
        successful_tasks = [t for t in recent_tasks if t.get('success', True)]
        failed_tasks = [t for t in recent_tasks if not t.get('success', True)]
        
        avg_duration = 0.0
        if successful_tasks:
            durations = [t.get('duration', 0) for t in successful_tasks if t.get('duration')]
            if durations:
                avg_duration = statistics.mean(durations)
        
        return {
            'metrics': {
                'total_tasks_hour': len(recent_tasks),
                'successful_tasks_hour': len(successful_tasks),
                'failed_tasks_hour': len(failed_tasks),
                'success_rate': len(successful_tasks) / max(1, len(recent_tasks)) * 100,
                'avg_task_duration': avg_duration,
                'tasks_per_minute': len(recent_tasks) / 60.0
            }
        }
    
    def _get_task_history(self, limit: int) -> Dict:
        """Get recent task history."""
        recent_tasks = self.task_history[-limit:] if limit > 0 else self.task_history
        return {'tasks': recent_tasks}
    
    async def _update_metrics(self) -> None:
        """Update cluster metrics periodically."""
        while self.running:
            try:
                # Update cluster metrics
                status = self._get_cluster_status()
                self.cluster_metrics.total_nodes = status['total_nodes']
                self.cluster_metrics.active_nodes = status['active_nodes']
                self.cluster_metrics.cluster_health = status['cluster_health']
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
    
    async def _log_periodic_status(self) -> None:
        """Log periodic status updates."""
        while self.running:
            try:
                status = self._get_cluster_status()
                self.logger.info(
                    f"Cluster Status: {status['active_nodes']}/{status['total_nodes']} nodes active, "
                    f"{status['cluster_health']:.1f}% healthy, "
                    f"{status['successful_tasks']} successful tasks, "
                    f"{status['failed_tasks']} failed tasks"
                )
                
                await asyncio.sleep(30)  # Log every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error logging status: {e}")


async def main():
    """Run the cluster monitor."""
    coordinator_host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    
    monitor = ClusterMonitor(coordinator_host=coordinator_host)
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
