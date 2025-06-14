#!/usr/bin/env python3
"""
Load Test Metrics Exporter for Prometheus
Emits order_req_rate, order_err_rate, latency_ms during load testing
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry, write_to_textfile
import threading
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [METRICS] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadMetrics:
    """Load test metrics structure"""
    order_req_rate: float = 0.0
    order_err_rate: float = 0.0
    latency_ms: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timestamp: Optional[datetime] = None


class LoadMetricsExporter:
    """Prometheus metrics exporter for load testing"""
    
    def __init__(self, metrics_file: str = "load_test_metrics.prom", registry: Optional[CollectorRegistry] = None):
        self.metrics_file = Path(metrics_file)
        self.registry = registry or CollectorRegistry()
        
        # Initialize Prometheus metrics
        self._init_metrics()
        
        # State tracking
        self.running = False
        self.update_thread = None
        self.last_metrics = LoadMetrics()
        
        logger.info(f"Load metrics exporter initialized, output: {self.metrics_file}")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        # Request rate gauge
        self.order_req_rate_gauge = Gauge(
            'load_test_order_req_rate',
            'Current order request rate (req/sec)',
            registry=self.registry
        )
        
        # Error rate gauge
        self.order_err_rate_gauge = Gauge(
            'load_test_order_err_rate',
            'Current order error rate (percentage)',
            registry=self.registry
        )
        
        # Latency gauge
        self.latency_ms_gauge = Gauge(
            'load_test_latency_ms',
            'Average response latency (milliseconds)',
            registry=self.registry
        )
        
        # Total counters
        self.total_requests_counter = Counter(
            'load_test_total_requests',
            'Total number of requests sent',
            registry=self.registry
        )
        
        self.successful_requests_counter = Counter(
            'load_test_successful_requests',
            'Total number of successful requests',
            registry=self.registry
        )
        
        self.failed_requests_counter = Counter(
            'load_test_failed_requests',
            'Total number of failed requests',
            registry=self.registry
        )
        
        # Latency histogram
        self.latency_histogram = Histogram(
            'load_test_request_duration_seconds',
            'Request duration in seconds',
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage_gauge = Gauge(
            'load_test_cpu_usage_percent',
            'CPU usage percentage during load test',
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'load_test_memory_usage_percent',
            'Memory usage percentage during load test',
            registry=self.registry
        )
    
    def update_metrics(self, metrics: LoadMetrics):
        """Update metrics with new values"""
        try:
            # Update gauges
            self.order_req_rate_gauge.set(metrics.order_req_rate)
            self.order_err_rate_gauge.set(metrics.order_err_rate)
            self.latency_ms_gauge.set(metrics.latency_ms)
            
            # Update counters (only increment by difference)
            total_diff = metrics.total_requests - self.last_metrics.total_requests
            success_diff = metrics.successful_requests - self.last_metrics.successful_requests
            failed_diff = metrics.failed_requests - self.last_metrics.failed_requests
            
            if total_diff > 0:
                self.total_requests_counter._value._value += total_diff
            if success_diff > 0:
                self.successful_requests_counter._value._value += success_diff
            if failed_diff > 0:
                self.failed_requests_counter._value._value += failed_diff
            
            # Record latency in histogram (convert ms to seconds)
            if metrics.latency_ms > 0:
                self.latency_histogram.observe(metrics.latency_ms / 1000.0)
            
            # Update last metrics
            self.last_metrics = metrics
            
            logger.debug(f"Metrics updated: {metrics.order_req_rate:.1f} req/s, {metrics.order_err_rate:.2f}% errors")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")
    
    def export_to_file(self):
        """Export metrics to Prometheus format file"""
        try:
            write_to_textfile(str(self.metrics_file), self.registry)
            logger.debug(f"Metrics exported to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to export metrics to file: {str(e)}")
    
    def load_metrics_from_json(self, json_file: str) -> Optional[LoadMetrics]:
        """Load metrics from JSON file (from load runner)"""
        try:
            json_path = Path(json_file)
            if not json_path.exists():
                return None
                
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract metrics from load runner format
            if 'metrics' in data:
                metrics_data = data['metrics']
                return LoadMetrics(
                    order_req_rate=metrics_data.get('current_rate', 0.0),
                    order_err_rate=metrics_data.get('error_rate', 0.0) * 100,  # Convert to percentage
                    latency_ms=metrics_data.get('avg_latency', 0.0) * 1000,  # Convert to ms
                    total_requests=metrics_data.get('total_requests', 0),
                    successful_requests=metrics_data.get('successful_requests', 0),
                    failed_requests=metrics_data.get('failed_requests', 0),
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load metrics from JSON: {str(e)}")
            return None
    
    def monitor_load_runner(self, poll_interval: int = 15):
        """Monitor load runner metrics and export to Prometheus"""
        logger.info(f"Starting load metrics monitoring (poll interval: {poll_interval}s)")
        
        while self.running:
            try:
                # Look for latest load test results
                results_pattern = "load_test_results_*.json"
                results_files = list(Path(".").glob(results_pattern))
                
                if results_files:
                    # Get the most recent file
                    latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
                    
                    # Load metrics
                    metrics = self.load_metrics_from_json(str(latest_file))
                    if metrics:
                        self.update_metrics(metrics)
                        self.export_to_file()
                
                # Also check for simple metrics file
                simple_metrics_file = Path("load_test_metrics.prom")
                if simple_metrics_file.exists():
                    try:
                        with open(simple_metrics_file, 'r') as f:
                            content = f.read()
                            
                        # Parse simple Prometheus format
                        lines = content.strip().split('\n')
                        metrics_dict = {}
                        
                        for line in lines:
                            if line.startswith('#') or not line.strip():
                                continue
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                metric_name, value = parts
                                try:
                                    metrics_dict[metric_name] = float(value)
                                except ValueError:
                                    continue
                        
                        # Update metrics from parsed data
                        if metrics_dict:
                            metrics = LoadMetrics(
                                order_req_rate=metrics_dict.get('order_req_rate', 0.0),
                                order_err_rate=metrics_dict.get('order_err_rate', 0.0),
                                latency_ms=metrics_dict.get('latency_ms', 0.0),
                                total_requests=int(metrics_dict.get('total_requests', 0)),
                                successful_requests=int(metrics_dict.get('successful_requests', 0)),
                                failed_requests=int(metrics_dict.get('failed_requests', 0)),
                                timestamp=datetime.now()
                            )
                            
                            self.update_metrics(metrics)
                            self.export_to_file()
                    
                    except Exception as e:
                        logger.debug(f"Failed to parse simple metrics file: {str(e)}")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(poll_interval)
        
        logger.info("Load metrics monitoring stopped")
    
    def start_monitoring(self, poll_interval: int = 15):
        """Start background monitoring thread"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(
            target=self.monitor_load_runner,
            args=(poll_interval,),
            daemon=True
        )
        self.update_thread.start()
        logger.info("Load metrics monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Load metrics monitoring stopped")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Load test metrics exporter')
    parser.add_argument('--output', default='load_test_metrics.prom',
                       help='Output file for Prometheus metrics')
    parser.add_argument('--poll-interval', type=int, default=15,
                       help='Polling interval in seconds')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    # Create exporter
    with LoadMetricsExporter(args.output) as exporter:
        try:
            # Start monitoring
            exporter.start_monitoring(args.poll_interval)
            
            # Run for specified duration
            logger.info(f"Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
            
            logger.info("Monitoring completed")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Monitoring failed: {str(e)}")
            return 1


if __name__ == "__main__":
    sys.exit(main())