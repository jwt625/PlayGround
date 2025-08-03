#!/usr/bin/env python3
"""
Bay Bridge Traffic Detection System - Main Entry Point

This is the main entry point for the Bay Bridge traffic detection system with
integrated Prometheus metrics collection and Grafana Cloud monitoring.

Features:
- Real-time motion detection and vehicle tracking
- Traffic counting with directional analysis
- Prometheus metrics collection and HTTP server
- System health monitoring
- Performance optimization for 30+ FPS processing

Usage:
    python main.py [--config CONFIG_FILE] [--debug] [--no-metrics]

The system will:
1. Initialize webcam capture from Bay Bridge traffic cam
2. Start motion detection and object tracking
3. Begin traffic counting with directional analysis
4. Start metrics collection and HTTP server (port 9091)
5. Monitor system health and performance

Metrics are exposed at http://localhost:9091/metrics for Prometheus scraping.
"""

import os
import sys
import time
import signal
import argparse
import logging
from typing import Optional
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import system components
from motion_detector import MotionDetector
from object_tracker import ObjectTracker
from prometheus_metrics import initialize_metrics, get_metrics, shutdown_metrics, MetricsConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficDetectionSystem:
    """
    Main traffic detection system with integrated monitoring.

    Coordinates motion detection, object tracking, traffic counting,
    and metrics collection for comprehensive traffic monitoring.
    """

    def __init__(self, config_file: Optional[str] = None, enable_metrics: bool = True, debug: bool = False):
        self.config_file = config_file
        self.enable_metrics = enable_metrics
        self.debug = debug
        self.running = False

        # System components
        self.motion_detector: Optional[MotionDetector] = None
        self.object_tracker: Optional[ObjectTracker] = None
        self.metrics = None

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0.0

        logger.info("TrafficDetectionSystem initialized")

    def initialize_components(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing system components...")

            # Initialize metrics system first
            if self.enable_metrics:
                logger.info("Initializing metrics collection...")
                config = MetricsConfig.from_env()
                config.debug = self.debug
                self.metrics = initialize_metrics(config)
                logger.info(f"Metrics server started on port {config.http_server_port}")

            # Initialize motion detector
            logger.info("Initializing motion detector...")
            self.motion_detector = MotionDetector()

            # Initialize object tracker
            logger.info("Initializing object tracker...")
            self.object_tracker = ObjectTracker()

            # Update system status
            if self.metrics:
                self.metrics.update_system_status('webcam', True)
                self.metrics.update_system_status('detector', True)
                self.metrics.update_system_status('tracker', True)

            logger.info("✅ All components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            if self.metrics:
                self.metrics.update_system_status('webcam', False)
                self.metrics.update_system_status('detector', False)
                self.metrics.update_system_status('tracker', False)
            raise

    def process_frame(self, frame):
        """
        Process a single frame through the detection pipeline.

        Args:
            frame: Input video frame

        Returns:
            Processed frame with annotations
        """
        frame_start_time = time.time()

        try:
            # Motion detection
            motion_result = self.motion_detector.detect_motion(frame)

            # Object tracking and traffic counting
            if motion_result and hasattr(motion_result, 'contours'):
                tracking_result = self.object_tracker.update(motion_result.contours, frame)

                # Record traffic counts in metrics
                if self.metrics and hasattr(tracking_result, 'new_counts'):
                    for direction, count in tracking_result.new_counts.items():
                        for _ in range(count):
                            self.metrics.record_vehicle_count(direction)

            # Update performance metrics
            frame_processing_time = time.time() - frame_start_time
            self.frame_count += 1

            # Calculate FPS every second
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                elapsed = current_time - self.last_fps_update
                self.fps = self.frame_count / (current_time - self.start_time)
                self.last_fps_update = current_time

                # Update metrics
                if self.metrics:
                    active_objects = len(self.object_tracker.tracked_objects) if self.object_tracker else 0
                    self.metrics.update_performance_metrics(
                        fps=self.fps,
                        active_objects=active_objects,
                        processing_time=frame_processing_time
                    )

                if self.debug:
                    logger.debug(f"FPS: {self.fps:.1f}, Processing: {frame_processing_time*1000:.1f}ms")

            return motion_result.annotated_frame if motion_result else frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            if self.metrics:
                self.metrics.update_system_status('detector', False)
            return frame

    def run(self):
        """Run the main detection loop."""
        try:
            logger.info("🚀 Starting Bay Bridge Traffic Detection System")

            # Initialize all components
            self.initialize_components()

            # Start main processing loop
            self.running = True
            logger.info("📹 Starting video processing...")
            logger.info("📊 Metrics available at: http://localhost:9091/metrics")
            logger.info("🛑 Press Ctrl+C to stop")

            # Main processing loop
            while self.running:
                try:
                    # Get frame from motion detector
                    if self.motion_detector and hasattr(self.motion_detector, 'get_frame'):
                        frame = self.motion_detector.get_frame()
                        if frame is not None:
                            processed_frame = self.process_frame(frame)

                            # Display frame (optional)
                            if self.debug:
                                import cv2
                                cv2.imshow('Bay Bridge Traffic Detection', processed_frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                    else:
                        # Fallback: simulate processing for testing
                        time.sleep(0.033)  # ~30 FPS

                except KeyboardInterrupt:
                    logger.info("🛑 Shutdown requested by user")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    if self.metrics:
                        self.metrics.update_system_status('detector', False)
                    time.sleep(1)  # Brief pause before retrying

        except Exception as e:
            logger.error(f"❌ System error: {e}")
            raise
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown all system components gracefully."""
        logger.info("🔄 Shutting down system...")
        self.running = False

        try:
            # Shutdown motion detector
            if self.motion_detector and hasattr(self.motion_detector, 'cleanup'):
                self.motion_detector.cleanup()

            # Shutdown object tracker
            if self.object_tracker and hasattr(self.object_tracker, 'cleanup'):
                self.object_tracker.cleanup()

            # Shutdown metrics
            if self.enable_metrics:
                shutdown_metrics()

            # Close OpenCV windows
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass

            logger.info("✅ System shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def setup_signal_handlers(system: TrafficDetectionSystem):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}")
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bay Bridge Traffic Detection System with Prometheus Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default settings
  python main.py --debug                   # Start with debug output
  python main.py --no-metrics              # Start without metrics collection
  python main.py --config custom.env      # Use custom configuration file

The system will start traffic detection and expose metrics at:
  http://localhost:9091/metrics

Configure Prometheus to scrape this endpoint for monitoring.
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path (default: .env)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output and visualization'
    )

    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Disable metrics collection'
    )

    args = parser.parse_args()

    # Load custom config file if specified
    if args.config:
        if os.path.exists(args.config):
            load_dotenv(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

    # Create and run system
    try:
        system = TrafficDetectionSystem(
            config_file=args.config,
            enable_metrics=not args.no_metrics,
            debug=args.debug
        )

        setup_signal_handlers(system)
        system.run()

    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
