#!/usr/bin/env python3
"""
Auto-Shutdown Cost Optimizer
Phase P11 Week 4 Day 3 - Automated resource scaling based on activity

Features:
- Monitor system load and automatically scale down idle resources
- Cost tracking with AWS Cost Explorer integration
- Prometheus metrics for cost monitoring
- Configurable thresholds and grace periods
- Safe shutdown with state preservation
"""

import os
import time
import logging
import asyncio
import subprocess
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import psutil
import requests

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    AWS_AVAILABLE = False

from prometheus_client import Gauge, Counter, Histogram, start_http_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('auto_shutdown')

# Prometheus metrics
shutdown_events = Counter('auto_shutdown_events_total', 'Total shutdown events', ['resource_type', 'reason'])
startup_events = Counter('auto_startup_events_total', 'Total startup events', ['resource_type', 'trigger'])
resource_idle_time = Gauge('resource_idle_time_seconds', 'Resource idle time in seconds', ['resource_name'])
cost_savings_usd = Gauge('cost_savings_usd_total', 'Total cost savings in USD', ['period'])
current_cost_rate = Gauge('current_cost_rate_usd_per_hour', 'Current cost rate in USD per hour')
system_load_avg = Gauge('system_load_average', 'System load average', ['period'])


@dataclass
class ResourceConfig:
    """Configuration for a resource to monitor"""
    name: str
    type: str  # 'deployment', 'statefulset', 'ec2', 'rds'
    namespace: str = 'default'
    min_replicas: int = 0
    max_replicas: int = 10
    idle_threshold_minutes: int = 30
    cpu_threshold_percent: float = 5.0
    memory_threshold_percent: float = 10.0
    cost_per_hour_usd: float = 0.0
    shutdown_command: Optional[str] = None
    startup_command: Optional[str] = None


@dataclass
class AutoShutdownConfig:
    """Main configuration for auto-shutdown system"""
    check_interval_minutes: int = 5
    idle_threshold_minutes: int = 30
    enable_shutdown: bool = True
    enable_startup: bool = True
    dry_run: bool = False
    
    # Safety settings
    min_business_hours: int = 9    # 9 AM
    max_business_hours: int = 17   # 5 PM
    weekend_shutdown_enabled: bool = True
    
    # AWS settings
    aws_region: str = 'us-east-1'
    cost_explorer_enabled: bool = True
    
    # Prometheus settings
    metrics_port: int = 9091


class SystemMonitor:
    """Monitor system resource usage and activity"""
    
    def __init__(self):
        self.history = []
        self.max_history = 100
    
    def get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics"""
        load_avg = psutil.getloadavg()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            'load_1min': load_avg[0],
            'load_5min': load_avg[1],
            'load_15min': load_avg[2],
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'timestamp': time.time()
        }
        
        # Update Prometheus metrics
        system_load_avg.labels(period='1min').set(load_avg[0])
        system_load_avg.labels(period='5min').set(load_avg[1])
        system_load_avg.labels(period='15min').set(load_avg[2])
        
        # Store in history
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return metrics
    
    def get_network_activity(self) -> Dict[str, int]:
        """Get network I/O activity"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def is_system_idle(self, cpu_threshold: float = 5.0, memory_threshold: float = 10.0) -> bool:
        """Check if system is currently idle"""
        if len(self.history) < 3:
            return False
        
        # Check last 3 measurements
        recent_metrics = self.history[-3:]
        
        for metrics in recent_metrics:
            if metrics['cpu_percent'] > cpu_threshold:
                return False
            if metrics['memory_percent'] > memory_threshold:
                return False
        
        return True
    
    def get_idle_duration(self, cpu_threshold: float = 5.0, memory_threshold: float = 10.0) -> float:
        """Get duration system has been idle (in minutes)"""
        if not self.is_system_idle(cpu_threshold, memory_threshold):
            return 0.0
        
        # Find when system became idle
        idle_start = None
        current_time = time.time()
        
        for metrics in reversed(self.history):
            if metrics['cpu_percent'] > cpu_threshold or metrics['memory_percent'] > memory_threshold:
                break
            idle_start = metrics['timestamp']
        
        if idle_start is None:
            return 0.0
        
        return (current_time - idle_start) / 60.0  # Convert to minutes


class KubernetesManager:
    """Manage Kubernetes resources"""
    
    def __init__(self):
        self.kubectl_available = self._check_kubectl()
    
    def _check_kubectl(self) -> bool:
        """Check if kubectl is available"""
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_deployment_info(self, name: str, namespace: str = 'default') -> Optional[Dict]:
        """Get deployment information"""
        if not self.kubectl_available:
            return None
        
        try:
            result = subprocess.run([
                'kubectl', 'get', 'deployment', name, '-n', namespace, 
                '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Failed to get deployment info for {name}: {e}")
        
        return None
    
    def scale_deployment(self, name: str, replicas: int, namespace: str = 'default') -> bool:
        """Scale deployment to specified replicas"""
        if not self.kubectl_available:
            logger.error("kubectl not available")
            return False
        
        try:
            result = subprocess.run([
                'kubectl', 'scale', 'deployment', name, 
                f'--replicas={replicas}', '-n', namespace
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Scaled deployment {name} to {replicas} replicas")
                return True
            else:
                logger.error(f"Failed to scale {name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error scaling deployment {name}: {e}")
            return False
    
    def get_pod_metrics(self, deployment: str, namespace: str = 'default') -> List[Dict]:
        """Get resource metrics for pods in deployment"""
        if not self.kubectl_available:
            return []
        
        try:
            # Get pods for deployment
            result = subprocess.run([
                'kubectl', 'get', 'pods', '-n', namespace,
                '-l', f'app={deployment}', '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return []
            
            pods_data = json.loads(result.stdout)
            metrics = []
            
            for pod in pods_data.get('items', []):
                pod_name = pod['metadata']['name']
                
                # Get metrics if metrics server is available
                metrics_result = subprocess.run([
                    'kubectl', 'top', 'pod', pod_name, '-n', namespace,
                    '--no-headers'
                ], capture_output=True, text=True, timeout=10)
                
                if metrics_result.returncode == 0:
                    # Parse metrics output: NAME CPU(cores) MEMORY(bytes)
                    lines = metrics_result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            metrics.append({
                                'pod_name': parts[0],
                                'cpu': parts[1],
                                'memory': parts[2]
                            })
            
            return metrics
        except Exception as e:
            logger.warning(f"Failed to get pod metrics: {e}")
            return []


class AWSCostManager:
    """Manage AWS cost tracking and optimization"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.cost_explorer = None
        self.ec2 = None
        
        if AWS_AVAILABLE:
            try:
                self.cost_explorer = boto3.client('ce', region_name=region)
                self.ec2 = boto3.client('ec2', region_name=region)
            except Exception as e:
                logger.warning(f"Failed to initialize AWS clients: {e}")
    
    def get_daily_costs(self, days: int = 7) -> Dict[str, float]:
        """Get daily costs for the past N days"""
        if not self.cost_explorer:
            return {}
        
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            daily_costs = {}
            for result in response.get('ResultsByTime', []):
                date = result['TimePeriod']['Start']
                total_cost = 0.0
                
                for group in result.get('Groups', []):
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    total_cost += cost
                
                daily_costs[date] = total_cost
            
            return daily_costs
        except Exception as e:
            logger.error(f"Failed to get daily costs: {e}")
            return {}
    
    def get_current_monthly_cost(self) -> float:
        """Get current month-to-date cost"""
        if not self.cost_explorer:
            return 0.0
        
        try:
            now = datetime.now()
            start_date = now.replace(day=1).date()
            end_date = now.date()
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['BlendedCost']
            )
            
            for result in response.get('ResultsByTime', []):
                return float(result['Total']['BlendedCost']['Amount'])
            
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get monthly cost: {e}")
            return 0.0
    
    def stop_ec2_instances(self, instance_ids: List[str]) -> Dict[str, bool]:
        """Stop EC2 instances"""
        if not self.ec2:
            return {}
        
        results = {}
        
        for instance_id in instance_ids:
            try:
                response = self.ec2.stop_instances(InstanceIds=[instance_id])
                results[instance_id] = True
                logger.info(f"Stopping EC2 instance: {instance_id}")
            except Exception as e:
                logger.error(f"Failed to stop instance {instance_id}: {e}")
                results[instance_id] = False
        
        return results
    
    def start_ec2_instances(self, instance_ids: List[str]) -> Dict[str, bool]:
        """Start EC2 instances"""
        if not self.ec2:
            return {}
        
        results = {}
        
        for instance_id in instance_ids:
            try:
                response = self.ec2.start_instances(InstanceIds=[instance_id])
                results[instance_id] = True
                logger.info(f"Starting EC2 instance: {instance_id}")
            except Exception as e:
                logger.error(f"Failed to start instance {instance_id}: {e}")
                results[instance_id] = False
        
        return results


class AutoShutdownManager:
    """Main auto-shutdown manager"""
    
    def __init__(self, config: AutoShutdownConfig):
        self.config = config
        self.system_monitor = SystemMonitor()
        self.k8s_manager = KubernetesManager()
        self.aws_manager = AWSCostManager(config.aws_region) if AWS_AVAILABLE else None
        
        # Resource configurations
        self.resources = []
        self.shutdown_state = {}
        self.cost_savings = 0.0
        
        # State file for persistence
        self.state_file = Path('/tmp/auto_shutdown_state.json')
        self.load_state()
    
    def add_resource(self, resource: ResourceConfig):
        """Add resource to monitor"""
        self.resources.append(resource)
        logger.info(f"Added resource: {resource.name} ({resource.type})")
    
    def load_state(self):
        """Load shutdown state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.shutdown_state = data.get('shutdown_state', {})
                    self.cost_savings = data.get('cost_savings', 0.0)
                logger.info(f"Loaded state: {len(self.shutdown_state)} resources tracked")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    def save_state(self):
        """Save shutdown state to file"""
        try:
            data = {
                'shutdown_state': self.shutdown_state,
                'cost_savings': self.cost_savings,
                'last_updated': time.time()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def is_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        now = datetime.now()
        
        # Check weekend
        if now.weekday() >= 5 and not self.config.weekend_shutdown_enabled:
            return True  # Treat weekend as business hours if weekend shutdown disabled
        
        # Check time
        hour = now.hour
        return self.config.min_business_hours <= hour < self.config.max_business_hours
    
    def should_shutdown_resource(self, resource: ResourceConfig) -> Tuple[bool, str]:
        """Determine if resource should be shut down"""
        # Check if already shut down
        if self.shutdown_state.get(resource.name, {}).get('is_shutdown', False):
            return False, "already_shutdown"
        
        # Check business hours (only shutdown outside business hours)
        if self.is_business_hours():
            return False, "business_hours"
        
        # Check system load
        idle_duration = self.system_monitor.get_idle_duration(
            resource.cpu_threshold_percent,
            resource.memory_threshold_percent
        )
        
        if idle_duration < resource.idle_threshold_minutes:
            return False, f"not_idle_enough_{idle_duration:.1f}min"
        
        # Update idle time metric
        resource_idle_time.labels(resource_name=resource.name).set(idle_duration * 60)
        
        return True, f"idle_{idle_duration:.1f}min"
    
    def should_startup_resource(self, resource: ResourceConfig) -> Tuple[bool, str]:
        """Determine if resource should be started up"""
        shutdown_info = self.shutdown_state.get(resource.name, {})
        
        # Only consider if currently shut down
        if not shutdown_info.get('is_shutdown', False):
            return False, "not_shutdown"
        
        # Check if business hours started
        if self.is_business_hours():
            return True, "business_hours_started"
        
        # Check if load increased
        current_load = self.system_monitor.get_system_load()
        if (current_load['cpu_percent'] > resource.cpu_threshold_percent or 
            current_load['memory_percent'] > resource.memory_threshold_percent):
            return True, "load_increased"
        
        return False, "no_trigger"
    
    def shutdown_resource(self, resource: ResourceConfig, reason: str) -> bool:
        """Shutdown a specific resource"""
        logger.info(f"Shutting down {resource.name} ({resource.type}): {reason}")
        
        if self.config.dry_run:
            logger.info(f"DRY RUN: Would shutdown {resource.name}")
            return True
        
        success = False
        
        try:
            if resource.type == 'deployment':
                # Get current replica count
                deployment_info = self.k8s_manager.get_deployment_info(resource.name, resource.namespace)
                if deployment_info:
                    current_replicas = deployment_info['spec']['replicas']
                    
                    # Scale to minimum replicas
                    success = self.k8s_manager.scale_deployment(
                        resource.name, resource.min_replicas, resource.namespace
                    )
                    
                    if success:
                        self.shutdown_state[resource.name] = {
                            'is_shutdown': True,
                            'shutdown_time': time.time(),
                            'original_replicas': current_replicas,
                            'reason': reason,
                            'type': resource.type
                        }
            
            elif resource.shutdown_command:
                # Execute custom shutdown command
                result = subprocess.run(
                    resource.shutdown_command.split(),
                    capture_output=True, text=True, timeout=300
                )
                success = result.returncode == 0
                
                if success:
                    self.shutdown_state[resource.name] = {
                        'is_shutdown': True,
                        'shutdown_time': time.time(),
                        'reason': reason,
                        'type': resource.type
                    }
            
            if success:
                # Update metrics
                shutdown_events.labels(resource_type=resource.type, reason=reason).inc()
                
                # Calculate cost savings
                self.cost_savings += resource.cost_per_hour_usd
                cost_savings_usd.labels(period='total').set(self.cost_savings)
                
                # Save state
                self.save_state()
                
                logger.info(f"Successfully shut down {resource.name}")
            else:
                logger.error(f"Failed to shutdown {resource.name}")
        
        except Exception as e:
            logger.error(f"Error shutting down {resource.name}: {e}")
            success = False
        
        return success
    
    def startup_resource(self, resource: ResourceConfig, trigger: str) -> bool:
        """Start up a specific resource"""
        shutdown_info = self.shutdown_state.get(resource.name, {})
        
        if not shutdown_info.get('is_shutdown', False):
            return False
        
        logger.info(f"Starting up {resource.name} ({resource.type}): {trigger}")
        
        if self.config.dry_run:
            logger.info(f"DRY RUN: Would startup {resource.name}")
            return True
        
        success = False
        
        try:
            if resource.type == 'deployment':
                # Restore original replica count
                original_replicas = shutdown_info.get('original_replicas', resource.max_replicas)
                success = self.k8s_manager.scale_deployment(
                    resource.name, original_replicas, resource.namespace
                )
            
            elif resource.startup_command:
                # Execute custom startup command
                result = subprocess.run(
                    resource.startup_command.split(),
                    capture_output=True, text=True, timeout=300
                )
                success = result.returncode == 0
            
            if success:
                # Update state
                del self.shutdown_state[resource.name]
                
                # Update metrics
                startup_events.labels(resource_type=resource.type, trigger=trigger).inc()
                
                # Save state
                self.save_state()
                
                logger.info(f"Successfully started up {resource.name}")
            else:
                logger.error(f"Failed to startup {resource.name}")
        
        except Exception as e:
            logger.error(f"Error starting up {resource.name}: {e}")
            success = False
        
        return success
    
    def check_resources(self):
        """Check all resources and perform shutdown/startup as needed"""
        logger.debug("Checking resources for auto-shutdown/startup")
        
        # Update system metrics
        self.system_monitor.get_system_load()
        
        for resource in self.resources:
            try:
                # Check for shutdown
                if self.config.enable_shutdown:
                    should_shutdown, shutdown_reason = self.should_shutdown_resource(resource)
                    if should_shutdown:
                        self.shutdown_resource(resource, shutdown_reason)
                
                # Check for startup
                if self.config.enable_startup:
                    should_startup, startup_reason = self.should_startup_resource(resource)
                    if should_startup:
                        self.startup_resource(resource, startup_reason)
                        
            except Exception as e:
                logger.error(f"Error checking resource {resource.name}: {e}")
        
        # Update cost metrics
        if self.aws_manager:
            try:
                monthly_cost = self.aws_manager.get_current_monthly_cost()
                current_cost_rate.set(monthly_cost / max(1, datetime.now().day))
            except Exception as e:
                logger.warning(f"Failed to update cost metrics: {e}")
    
    def run(self):
        """Run the auto-shutdown manager"""
        logger.info("Starting Auto-Shutdown Manager")
        logger.info(f"Check interval: {self.config.check_interval_minutes} minutes")
        logger.info(f"Idle threshold: {self.config.idle_threshold_minutes} minutes")
        logger.info(f"Resources monitored: {len(self.resources)}")
        logger.info(f"Dry run mode: {self.config.dry_run}")
        
        # Start Prometheus metrics server
        start_http_server(self.config.metrics_port)
        logger.info(f"Prometheus metrics server started on port {self.config.metrics_port}")
        
        try:
            while True:
                self.check_resources()
                time.sleep(self.config.check_interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Auto-shutdown manager stopped by user")
        except Exception as e:
            logger.error(f"Auto-shutdown manager error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Auto-Shutdown Cost Optimizer')
    parser.add_argument('--idle-min', type=int, default=30, 
                       help='Idle threshold in minutes (default: 30)')
    parser.add_argument('--check-interval', type=int, default=5,
                       help='Check interval in minutes (default: 5)')
    parser.add_argument('--scale', type=str, help='Scale resources: api=0,exec=0,dash=0')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--enable-weekend', action='store_true', 
                       help='Enable shutdown during weekends')
    parser.add_argument('--config', type=str, help='Configuration file')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AutoShutdownConfig(
        check_interval_minutes=args.check_interval,
        idle_threshold_minutes=args.idle_min,
        dry_run=args.dry_run,
        weekend_shutdown_enabled=args.enable_weekend
    )
    
    # Create manager
    manager = AutoShutdownManager(config)
    
    # Add default resources
    default_resources = [
        ResourceConfig(
            name='mech-exo-api',
            type='deployment',
            namespace='default',
            min_replicas=0,
            max_replicas=3,
            idle_threshold_minutes=args.idle_min,
            cost_per_hour_usd=0.15
        ),
        ResourceConfig(
            name='mech-exo-exec',
            type='deployment', 
            namespace='default',
            min_replicas=0,
            max_replicas=4,
            idle_threshold_minutes=args.idle_min,
            cost_per_hour_usd=0.20
        ),
        ResourceConfig(
            name='mech-exo-dash',
            type='deployment',
            namespace='default', 
            min_replicas=0,
            max_replicas=2,
            idle_threshold_minutes=args.idle_min,
            cost_per_hour_usd=0.10
        )
    ]
    
    for resource in default_resources:
        manager.add_resource(resource)
    
    # Handle manual scaling
    if args.scale:
        logger.info("Manual scaling requested")
        scale_commands = args.scale.split(',')
        
        for command in scale_commands:
            if '=' in command:
                resource_name, replicas = command.split('=')
                replicas = int(replicas)
                
                # Find matching resource
                for resource in manager.resources:
                    if resource.name.endswith(resource_name):
                        logger.info(f"Scaling {resource.name} to {replicas} replicas")
                        
                        if not args.dry_run:
                            success = manager.k8s_manager.scale_deployment(
                                resource.name, replicas, resource.namespace
                            )
                            if success:
                                logger.info(f"Successfully scaled {resource.name}")
                            else:
                                logger.error(f"Failed to scale {resource.name}")
                        break
        
        # Exit after manual scaling
        return
    
    # Run continuous monitoring
    manager.run()


if __name__ == "__main__":
    main()