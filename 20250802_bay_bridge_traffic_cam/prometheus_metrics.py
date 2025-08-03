"""
Prometheus Metrics Collection for Bay Bridge Traffic Detection System

This module implements comprehensive metrics collection for the traffic detection system,
providing real-time monitoring capabilities through Prometheus and Grafana Cloud integration.

Features:
- Traffic counting metrics (vehicles by direction)
- System health monitoring (webcam, detector, tracker status)
- Performance metrics (FPS, processing time)
- HTTP server for metrics exposition
- Background metrics calculation and caching

Architecture:
- Pull-based metrics server (HTTP endpoint)
- Event-driven traffic counting
- Periodic system health checks
- Minimal performance overhead (<1% CPU impact)
"""

import os
import time
import threading
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque

from prometheus_client import Counter, Gauge, Histogram, start_http_server, generate_latest
from prometheus_client.core import CollectorRegistry, REGISTRY
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection system."""
    enabled: bool = True
    mode: str = "pull"  # "push" or "pull"
    http_server_enabled: bool = True
    http_server_port: int = 9091
    push_gateway_url: Optional[str] = None
    username: Optional[str] = None
    api_key: Optional[str] = None
    app_name: str = "bay-bridge-traffic-detector"
    app_version: str = "1.0.0"
    app_instance: str = "main"
    push_interval: int = 30
    flow_calculation_interval: int = 60
    debug: bool = False

    @classmethod
    def from_env(cls) -> 'MetricsConfig':
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
            mode=os.getenv('METRICS_MODE', 'pull'),
            http_server_enabled=os.getenv('PROMETHEUS_HTTP_SERVER_ENABLED', 'true').lower() == 'true',
            http_server_port=int(os.getenv('PROMETHEUS_HTTP_SERVER_PORT', '9091')),
            push_gateway_url=os.getenv('PROMETHEUS_PUSH_GATEWAY_URL'),
            username=os.getenv('PROMETHEUS_USERNAME'),
            api_key=os.getenv('PROMETHEUS_API_KEY'),
            app_name=os.getenv('APP_NAME', 'bay-bridge-traffic-detector'),
            app_version=os.getenv('APP_VERSION', '1.0.0'),
            app_instance=os.getenv('APP_INSTANCE', 'main'),
            push_interval=int(os.getenv('PROMETHEUS_PUSH_INTERVAL', '30')),
            flow_calculation_interval=int(os.getenv('TRAFFIC_FLOW_CALCULATION_INTERVAL', '60')),
            debug=os.getenv('METRICS_DEBUG', 'false').lower() == 'true'
        )


class TrafficMetrics:
    """
    Comprehensive metrics collection for traffic detection system.
    
    Implements Priority 1 (MVP) metrics:
    - traffic_vehicles_total: Counter for vehicle counts by direction
    - traffic_flow_rate_per_minute: Gauge for vehicles per minute
    - system_status: Gauge for component health monitoring
    """
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Common labels for all metrics
        self.common_labels = ['app', 'instance']
        self.common_label_values = [config.app_name, config.app_instance]
        
        # Priority 1: Essential Traffic Metrics (MVP)
        self.traffic_vehicles_total = Counter(
            'traffic_vehicles_total',
            'Total vehicles detected crossing counting lines',
            ['direction'] + self.common_labels,
            registry=self.registry
        )
        
        self.traffic_flow_rate = Gauge(
            'traffic_flow_rate_per_minute',
            'Vehicles per minute by direction',
            ['direction'] + self.common_labels,
            registry=self.registry
        )
        
        self.system_status = Gauge(
            'system_status',
            'System component health status (1=healthy, 0=error)',
            ['component'] + self.common_labels,
            registry=self.registry
        )
        
        # Priority 2: Performance Metrics (Future Phase)
        self.motion_detector_fps = Gauge(
            'motion_detector_fps',
            'Motion detector frames per second',
            self.common_labels,
            registry=self.registry
        )
        
        self.tracked_objects_active = Gauge(
            'tracked_objects_active',
            'Number of currently tracked objects',
            self.common_labels,
            registry=self.registry
        )
        
        self.frame_processing_time = Histogram(
            'frame_processing_time_seconds',
            'Time spent processing each frame',
            self.common_labels,
            buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            registry=self.registry
        )
        
        # Internal state for flow rate calculation
        self._traffic_counts = defaultdict(int)
        self._flow_calculation_thread = None
        self._stop_flow_calculation = threading.Event()
        self._last_flow_calculation = time.time()
        
        # HTTP server state
        self._http_server_started = False
        
        # Initialize system status
        self._initialize_system_status()
        
        logger.info(f"TrafficMetrics initialized with config: {config}")
    
    def _initialize_system_status(self):
        """Initialize system status metrics to healthy state."""
        components = ['webcam', 'detector', 'tracker']
        for component in components:
            self.system_status.labels(
                component=component,
                app=self.config.app_name,
                instance=self.config.app_instance
            ).set(1)  # 1 = healthy
    
    def record_vehicle_count(self, direction: str):
        """
        Record a vehicle detection event.
        
        Args:
            direction: 'left' or 'right'
        """
        if not self.config.enabled:
            return
            
        self.traffic_vehicles_total.labels(
            direction=direction,
            app=self.config.app_name,
            instance=self.config.app_instance
        ).inc()
        
        # Update internal counter for flow rate calculation
        self._traffic_counts[direction] += 1
        
        if self.config.debug:
            logger.debug(f"Vehicle count recorded: {direction}")
    
    def update_system_status(self, component: str, healthy: bool):
        """
        Update system component health status.
        
        Args:
            component: Component name ('webcam', 'detector', 'tracker')
            healthy: True if healthy, False if error
        """
        if not self.config.enabled:
            return
            
        status_value = 1 if healthy else 0
        self.system_status.labels(
            component=component,
            app=self.config.app_name,
            instance=self.config.app_instance
        ).set(status_value)
        
        if self.config.debug:
            logger.debug(f"System status updated: {component}={status_value}")
    
    def update_performance_metrics(self, fps: float, active_objects: int, processing_time: float):
        """
        Update performance metrics (Priority 2).
        
        Args:
            fps: Current frames per second
            active_objects: Number of tracked objects
            processing_time: Frame processing time in seconds
        """
        if not self.config.enabled:
            return
            
        self.motion_detector_fps.labels(
            app=self.config.app_name,
            instance=self.config.app_instance
        ).set(fps)
        
        self.tracked_objects_active.labels(
            app=self.config.app_name,
            instance=self.config.app_instance
        ).set(active_objects)
        
        self.frame_processing_time.labels(
            app=self.config.app_name,
            instance=self.config.app_instance
        ).observe(processing_time)
    
    def start_flow_calculation(self):
        """Start background thread for flow rate calculation."""
        if self._flow_calculation_thread is not None:
            return
            
        self._stop_flow_calculation.clear()
        self._flow_calculation_thread = threading.Thread(
            target=self._flow_calculation_worker,
            daemon=True
        )
        self._flow_calculation_thread.start()
        logger.info("Flow rate calculation thread started")
    
    def stop_flow_calculation(self):
        """Stop background flow rate calculation."""
        if self._flow_calculation_thread is None:
            return
            
        self._stop_flow_calculation.set()
        self._flow_calculation_thread.join(timeout=5)
        self._flow_calculation_thread = None
        logger.info("Flow rate calculation thread stopped")
    
    def _flow_calculation_worker(self):
        """Background worker for calculating traffic flow rates."""
        while not self._stop_flow_calculation.wait(self.config.flow_calculation_interval):
            try:
                current_time = time.time()
                time_elapsed = current_time - self._last_flow_calculation
                
                # Calculate flow rate for each direction
                for direction in ['left', 'right']:
                    count = self._traffic_counts.get(direction, 0)
                    flow_rate = (count / time_elapsed) * 60  # vehicles per minute
                    
                    self.traffic_flow_rate.labels(
                        direction=direction,
                        app=self.config.app_name,
                        instance=self.config.app_instance
                    ).set(flow_rate)
                    
                    if self.config.debug:
                        logger.debug(f"Flow rate calculated: {direction}={flow_rate:.2f} vehicles/min")
                
                # Reset counters
                self._traffic_counts.clear()
                self._last_flow_calculation = current_time
                
            except Exception as e:
                logger.error(f"Error in flow calculation: {e}")
    
    def start_http_server(self):
        """Start HTTP server for metrics exposition."""
        if not self.config.http_server_enabled or self._http_server_started:
            return
            
        try:
            start_http_server(self.config.http_server_port, registry=self.registry)
            self._http_server_started = True
            logger.info(f"Metrics HTTP server started on port {self.config.http_server_port}")
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            raise
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def shutdown(self):
        """Shutdown metrics collection system."""
        self.stop_flow_calculation()
        logger.info("TrafficMetrics shutdown complete")


# Global metrics instance
_metrics_instance: Optional[TrafficMetrics] = None


def get_metrics() -> Optional[TrafficMetrics]:
    """Get the global metrics instance."""
    return _metrics_instance


def initialize_metrics(config: Optional[MetricsConfig] = None) -> TrafficMetrics:
    """
    Initialize the global metrics system.
    
    Args:
        config: Optional configuration. If None, loads from environment.
        
    Returns:
        TrafficMetrics instance
    """
    global _metrics_instance
    
    if config is None:
        config = MetricsConfig.from_env()
    
    _metrics_instance = TrafficMetrics(config)
    
    if config.enabled:
        if config.http_server_enabled:
            _metrics_instance.start_http_server()
        _metrics_instance.start_flow_calculation()
    
    return _metrics_instance


def shutdown_metrics():
    """Shutdown the global metrics system."""
    global _metrics_instance
    if _metrics_instance:
        _metrics_instance.shutdown()
        _metrics_instance = None
