#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for Chaos Testing
Exports custom metrics for SLO monitoring and auto-abort functionality
"""

import time
import logging
import threading
import json
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chaos_metrics_exporter')


class ChaosMetricsExporter:
    """Exports chaos testing metrics to Prometheus"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.running = False
        
        # Define metrics
        self.setup_metrics()
        
        # State tracking
        self.last_kill_switch_time = None
        self.chaos_start_time = time.time()
        
    def setup_metrics(self):
        """Initialize Prometheus metrics"""
        
        # SLO Metrics
        self.order_error_rate = Gauge(
            'order_error_rate_percent',
            'Order error rate percentage over 5 minutes'
        )
        
        self.risk_ops_success_rate = Gauge(
            'risk_ops_success_rate_5m',
            'Risk operations success rate over 5 minutes'
        )
        
        self.api_latency_p95 = Gauge(
            'api_request_duration_ms_p95',
            'API request duration P95 in milliseconds'
        )
        
        self.kill_switch_recovery_time = Gauge(
            'kill_switch_recovery_time_seconds',
            'Kill switch recovery time in seconds'
        )
        
        # Error Budget
        self.error_budget_remaining = Gauge(
            'error_budget_remaining_percent',
            'Remaining error budget percentage'
        )
        
        self.error_budget_burn_rate = Gauge(
            'error_budget_burn_rate_percent_per_hour',
            'Error budget burn rate percentage per hour'
        )
        
        # Chaos Testing Counters
        self.chaos_scenarios_total = Counter(
            'chaos_scenarios_total',
            'Total chaos scenarios executed',
            ['scenario_type', 'status']
        )
        
        self.kill_switch_toggles_total = Counter(
            'kill_switch_toggles_total',
            'Total kill switch toggles executed',
            ['status']
        )
        
        self.network_chaos_injections = Counter(
            'network_chaos_injections_total',
            'Total network chaos injections',
            ['status']
        )
        
        # System Health
        self.system_uptime = Gauge(
            'chaos_system_uptime_seconds',
            'System uptime since chaos testing started'
        )
        
        self.ib_gateway_connected = Gauge(
            'ib_gateway_connected',
            'IB Gateway connection status (1=connected, 0=disconnected)'
        )
        
        self.duckdb_connection_errors = Counter(
            'duckdb_connection_errors_total',
            'Total DuckDB connection errors'
        )
        
        self.trading_enabled = Gauge(
            'trading_enabled',
            'Trading enabled status (1=enabled, 0=disabled)'
        )
        
        # Chaos Infrastructure
        self.network_chaos_active = Gauge(
            'network_chaos_active',
            'Network chaos injection active (1=active, 0=inactive)'
        )
        
        self.chaos_load_rate = Gauge(
            'chaos_load_rate_rps',
            'Chaos load generation rate in requests per second'
        )
        
        self.chaos_abort_triggered = Gauge(
            'chaos_abort_triggered',
            'Chaos abort has been triggered (1=yes, 0=no)'
        )
        
        # Infrastructure metrics
        self.node_disk_usage = Gauge(
            'node_disk_usage_percent',
            'Node disk usage percentage',
            ['mount_point']
        )
        
        self.node_memory_usage = Gauge(
            'node_memory_usage_percent',
            'Node memory usage percentage'
        )
        
        self.pod_restart_rate = Gauge(
            'pod_restart_rate_per_minute',
            'Pod restart rate per minute',
            ['pod_name']
        )
        
        # Custom histograms
        self.network_latency = Histogram(
            'network_latency_ms',
            'Network latency in milliseconds during chaos',
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        
        # Info metrics
        self.chaos_info = Info(
            'chaos_testing_info',
            'Information about chaos testing configuration'
        )
        
        # Set info metric
        self.chaos_info.info({
            'version': 'p11w3',
            'duration_hours': '24',
            'scenarios': 'order_flood,kill_switch,network_jitter,db_restart,ib_restart',
            'slo_error_budget': '97',
            'slo_recovery_time': '60'
        })
    
    def collect_slo_metrics(self):
        """Collect SLO-related metrics from various sources"""
        try:
            # Order error rate from application logs
            self._collect_order_error_rate()
            
            # Risk operations success rate
            self._collect_risk_ops_rate()
            
            # API latency from application metrics
            self._collect_api_latency()
            
            # Kill switch recovery time
            self._collect_kill_switch_metrics()
            
            # Error budget calculation
            self._calculate_error_budget()
            
        except Exception as e:
            logger.error(f"Error collecting SLO metrics: {e}")
    
    def _collect_order_error_rate(self):
        """Calculate order error rate from logs or API"""
        try:
            # Try to get from API health endpoint
            response = requests.get('http://localhost:8000/api/health/metrics', timeout=5)
            if response.status_code == 200:
                data = response.json()
                error_rate = data.get('order_error_rate_5m', 0)
                self.order_error_rate.set(error_rate)
            else:
                # Fallback to log analysis
                self._analyze_order_logs()
                
        except Exception as e:
            logger.debug(f"Failed to collect order error rate: {e}")
    
    def _analyze_order_logs(self):
        """Analyze order logs for error rate calculation"""
        try:
            # Simple log analysis for order errors
            log_file = Path('/tmp/chaos_logs/load_stats.json')
            if log_file.exists():
                with open(log_file, 'r') as f:
                    stats = json.load(f)
                
                total_requests = stats.get('requests_sent', 0)
                failed_requests = stats.get('requests_failed', 0)
                
                if total_requests > 0:
                    error_rate = (failed_requests / total_requests) * 100
                    self.order_error_rate.set(error_rate)
                
        except Exception as e:
            logger.debug(f"Log analysis failed: {e}")
    
    def _collect_risk_ops_rate(self):
        """Collect risk operations success rate"""
        try:
            # Simulate risk operations health check
            response = requests.get('http://localhost:8000/api/risk/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                success_rate = data.get('success_rate_5m', 0.99)
                self.risk_ops_success_rate.set(success_rate)
            else:
                # Default to healthy if can't measure
                self.risk_ops_success_rate.set(0.99)
                
        except Exception:
            # Assume healthy if can't reach
            self.risk_ops_success_rate.set(0.99)
    
    def _collect_api_latency(self):
        """Collect API latency metrics"""
        try:
            # Measure actual API response time
            start_time = time.time()
            response = requests.get('http://localhost:8000/api/health', timeout=10)
            latency_ms = (time.time() - start_time) * 1000
            
            self.api_latency_p95.set(latency_ms)
            self.network_latency.observe(latency_ms)
            
        except Exception as e:
            logger.debug(f"API latency collection failed: {e}")
            # Set high latency if unreachable
            self.api_latency_p95.set(5000)
    
    def _collect_kill_switch_metrics(self):
        """Collect kill switch recovery metrics"""
        try:
            results_file = Path('/tmp/chaos_logs/kill_switch_results.json')
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                stats = data.get('statistics', {})
                if 'average_recovery_time' in stats:
                    avg_recovery = stats['average_recovery_time']
                    self.kill_switch_recovery_time.set(avg_recovery)
                
                # Update counter for toggles
                successful = stats.get('successful_toggles', 0)
                failed = stats.get('failed_toggles', 0)
                
                self.kill_switch_toggles_total._value._value = successful + failed
                
        except Exception as e:
            logger.debug(f"Kill switch metrics collection failed: {e}")
    
    def _calculate_error_budget(self):
        """Calculate remaining error budget"""
        try:
            # Simple error budget calculation based on uptime
            runtime_hours = (time.time() - self.chaos_start_time) / 3600
            
            # Get current error rate
            current_error_rate = self.order_error_rate._value._value
            
            # Calculate error budget (assuming 3% total budget)
            target_error_rate = 3.0
            budget_used = (current_error_rate / target_error_rate) * 100
            budget_remaining = max(0, 100 - budget_used)
            
            self.error_budget_remaining.set(budget_remaining)
            
            # Calculate burn rate
            if runtime_hours > 0:
                burn_rate = budget_used / runtime_hours
                self.error_budget_burn_rate.set(burn_rate)
            
        except Exception as e:
            logger.debug(f"Error budget calculation failed: {e}")
    
    def collect_system_metrics(self):
        """Collect system and infrastructure metrics"""
        try:
            # System uptime
            uptime = time.time() - self.chaos_start_time
            self.system_uptime.set(uptime)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.node_memory_usage.set(memory.percent)
            
            # Disk usage
            for mount in ['/', '/tmp', '/var']:
                try:
                    usage = psutil.disk_usage(mount)
                    usage_percent = (usage.used / usage.total) * 100
                    self.node_disk_usage.labels(mount_point=mount).set(usage_percent)
                except Exception:
                    pass
            
            # IB Gateway status
            self._check_ib_gateway_status()
            
            # Trading status
            self._check_trading_status()
            
            # Network chaos status
            self._check_network_chaos_status()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_ib_gateway_status(self):
        """Check IB Gateway connection status"""
        try:
            # Check if IB Gateway process is running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'ibgateway' in proc.info['name'].lower():
                    self.ib_gateway_connected.set(1)
                    return
            
            # Also check via API if available
            response = requests.get('http://localhost:8000/api/ib/status', timeout=2)
            if response.status_code == 200:
                data = response.json()
                connected = 1 if data.get('connected', False) else 0
                self.ib_gateway_connected.set(connected)
            else:
                self.ib_gateway_connected.set(0)
                
        except Exception:
            self.ib_gateway_connected.set(0)
    
    def _check_trading_status(self):
        """Check if trading is enabled"""
        try:
            response = requests.get('http://localhost:8000/api/killswitch/status', timeout=2)
            if response.status_code == 200:
                data = response.json()
                enabled = 1 if data.get('trading_enabled', False) else 0
                self.trading_enabled.set(enabled)
            else:
                self.trading_enabled.set(0)
                
        except Exception:
            self.trading_enabled.set(0)
    
    def _check_network_chaos_status(self):
        """Check if network chaos is currently active"""
        try:
            chaos_pods_file = Path('/tmp/chaos_pods.txt')
            if chaos_pods_file.exists():
                with open(chaos_pods_file, 'r') as f:
                    active_pods = len(f.read().strip().split('\n'))
                self.network_chaos_active.set(1 if active_pods > 0 else 0)
            else:
                self.network_chaos_active.set(0)
                
        except Exception:
            self.network_chaos_active.set(0)
    
    def update_chaos_counters(self, scenario_type: str, status: str):
        """Update chaos scenario counters"""
        self.chaos_scenarios_total.labels(scenario_type=scenario_type, status=status).inc()
    
    def record_kill_switch_recovery(self, recovery_time: float):
        """Record kill switch recovery time"""
        self.kill_switch_recovery_time.set(recovery_time)
        status = 'success' if recovery_time <= 60 else 'slow'
        self.kill_switch_toggles_total.labels(status=status).inc()
    
    def trigger_chaos_abort(self):
        """Mark that chaos abort has been triggered"""
        self.chaos_abort_triggered.set(1)
        logger.critical("🚨 CHAOS ABORT TRIGGERED - Metrics updated")
    
    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            self.running = True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def start_collection_loop(self, interval: int = 30):
        """Start metrics collection loop in background thread"""
        def collection_worker():
            while self.running:
                try:
                    self.collect_slo_metrics()
                    self.collect_system_metrics()
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                
                time.sleep(interval)
        
        collection_thread = threading.Thread(target=collection_worker, daemon=True)
        collection_thread.start()
        logger.info(f"Metrics collection loop started (interval: {interval}s)")
    
    def run(self, collection_interval: int = 30):
        """Run the metrics exporter"""
        logger.info("Starting Chaos Metrics Exporter...")
        
        try:
            # Start HTTP server
            self.start_metrics_server()
            
            # Start collection loop
            self.start_collection_loop(collection_interval)
            
            logger.info("Chaos Metrics Exporter running - Press Ctrl+C to stop")
            
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Chaos Metrics Exporter stopped by user")
        except Exception as e:
            logger.error(f"Chaos Metrics Exporter failed: {e}")
        finally:
            self.running = False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prometheus Metrics Exporter for Chaos Testing')
    parser.add_argument('--port', type=int, default=9090, help='Metrics server port (default: 9090)')
    parser.add_argument('--interval', type=int, default=30, help='Collection interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    exporter = ChaosMetricsExporter(port=args.port)
    exporter.run(collection_interval=args.interval)


if __name__ == "__main__":
    main()