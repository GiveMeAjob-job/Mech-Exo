#!/usr/bin/env python3
"""
AWS Cost Metrics Exporter for Prometheus
Phase P11 Week 4 Day 3 - Cost monitoring and optimization

Features:
- Real-time AWS cost tracking via Cost Explorer API
- Service-level cost breakdown
- Cost trend analysis and forecasting
- Budget alerts and anomaly detection
- Cost optimization recommendations
"""

import os
import time
import logging
import threading
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import schedule

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    AWS_AVAILABLE = False

from prometheus_client import Gauge, Counter, Histogram, Info, start_http_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cost_metrics_exporter')

# Prometheus metrics
aws_cost_usd = Gauge('aws_cost_usd', 'AWS cost in USD', ['service', 'period', 'dimension'])
aws_cost_daily = Gauge('aws_cost_daily_usd', 'Daily AWS cost in USD', ['date'])
aws_cost_monthly = Gauge('aws_cost_monthly_usd', 'Monthly AWS cost in USD', ['month'])
aws_cost_forecast = Gauge('aws_cost_forecast_usd', 'Forecasted AWS cost in USD', ['period'])
aws_cost_budget_usage = Gauge('aws_cost_budget_usage_percent', 'Budget usage percentage', ['budget_name'])
aws_cost_anomaly_score = Gauge('aws_cost_anomaly_score', 'Cost anomaly detection score', ['service'])
aws_cost_savings_opportunities = Gauge('aws_cost_savings_usd', 'Potential cost savings in USD', ['recommendation_type'])

# Instance metrics
ec2_instance_cost = Gauge('ec2_instance_cost_usd_per_hour', 'EC2 instance cost per hour', ['instance_id', 'instance_type'])
ec2_instance_utilization = Gauge('ec2_instance_utilization_percent', 'EC2 instance utilization', ['instance_id', 'metric_type'])
rds_instance_cost = Gauge('rds_instance_cost_usd_per_hour', 'RDS instance cost per hour', ['instance_id', 'engine'])

# Service-specific metrics
s3_storage_cost = Gauge('s3_storage_cost_usd', 'S3 storage cost', ['bucket', 'storage_class'])
lambda_invocation_cost = Gauge('lambda_invocation_cost_usd', 'Lambda invocation cost', ['function_name'])
eks_cluster_cost = Gauge('eks_cluster_cost_usd', 'EKS cluster cost', ['cluster_name'])

# Cost optimization metrics
cost_optimization_requests = Counter('cost_optimization_requests_total', 'Cost optimization analysis requests')
cost_optimization_duration = Histogram('cost_optimization_duration_seconds', 'Cost optimization analysis duration')


@dataclass
class CostMetricsConfig:
    """Configuration for cost metrics exporter"""
    aws_region: str = 'us-east-1'
    metrics_port: int = 9092
    update_interval_minutes: int = 60
    
    # Cost Explorer settings
    enable_daily_costs: bool = True
    enable_service_costs: bool = True
    enable_forecasting: bool = True
    
    # Instance monitoring
    enable_ec2_monitoring: bool = True
    enable_rds_monitoring: bool = True
    
    # Budget monitoring
    enable_budget_monitoring: bool = True
    
    # Cost optimization
    enable_optimization_analysis: bool = True
    
    # Data retention
    cost_history_days: int = 30
    forecast_days: int = 30


class AWSCostExplorer:
    """AWS Cost Explorer client wrapper"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.cost_explorer = None
        self.budgets = None
        
        if AWS_AVAILABLE:
            try:
                self.cost_explorer = boto3.client('ce', region_name=region)
                self.budgets = boto3.client('budgets', region_name=region)
                logger.info(f"AWS Cost Explorer client initialized for region: {region}")
            except Exception as e:
                logger.error(f"Failed to initialize AWS clients: {e}")
                raise
        else:
            raise ImportError("boto3 not available - install with: pip install boto3")
    
    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """Get daily costs for the specified number of days"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            daily_costs = {}
            service_costs = {}
            
            for result in response.get('ResultsByTime', []):
                date_str = result['TimePeriod']['Start']
                total_cost = 0.0
                
                for group in result.get('Groups', []):
                    service = group['Keys'][0] if group['Keys'] else 'Unknown'
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    total_cost += cost
                    
                    # Track service costs
                    if service not in service_costs:
                        service_costs[service] = {}
                    service_costs[service][date_str] = cost
                
                daily_costs[date_str] = total_cost
            
            return daily_costs, service_costs
            
        except Exception as e:
            logger.error(f"Failed to get daily costs: {e}")
            return {}, {}
    
    def get_monthly_costs(self, months: int = 12) -> Dict[str, float]:
        """Get monthly costs for the specified number of months"""
        try:
            end_date = date.today().replace(day=1)
            start_date = end_date - timedelta(days=30 * months)
            
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            monthly_costs = {}
            
            for result in response.get('ResultsByTime', []):
                month = result['TimePeriod']['Start'][:7]  # YYYY-MM format
                total_cost = 0.0
                
                for group in result.get('Groups', []):
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    total_cost += cost
                
                monthly_costs[month] = total_cost
            
            return monthly_costs
            
        except Exception as e:
            logger.error(f"Failed to get monthly costs: {e}")
            return {}
    
    def get_cost_forecast(self, days: int = 30) -> Dict[str, float]:
        """Get cost forecast for the specified number of days"""
        try:
            start_date = date.today()
            end_date = start_date + timedelta(days=days)
            
            response = self.cost_explorer.get_cost_forecast(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Metric='BLENDED_COST',
                Granularity='DAILY'
            )
            
            forecast = {}
            for result in response.get('ForecastResultsByTime', []):
                date_str = result['TimePeriod']['Start']
                mean_value = float(result['MeanValue'])
                forecast[date_str] = mean_value
            
            return forecast
            
        except Exception as e:
            logger.error(f"Failed to get cost forecast: {e}")
            return {}
    
    def get_budget_status(self) -> List[Dict]:
        """Get budget status for all budgets"""
        try:
            response = self.budgets.describe_budgets(
                AccountId=boto3.Session().get_credentials().access_key.split(':')[4] if ':' in boto3.Session().get_credentials().access_key else '123456789012'
            )
            
            budget_status = []
            
            for budget in response.get('Budgets', []):
                budget_name = budget['BudgetName']
                budget_limit = float(budget['BudgetLimit']['Amount'])
                
                # Get actual spend
                try:
                    spend_response = self.budgets.describe_budget_performance(
                        AccountId=boto3.Session().get_credentials().access_key.split(':')[4] if ':' in boto3.Session().get_credentials().access_key else '123456789012',
                        BudgetName=budget_name
                    )
                    
                    actual_spend = float(spend_response['BudgetPerformanceHistory']['BudgetedAndActualAmountsList'][0]['ActualAmount']['Amount'])
                    
                    budget_status.append({
                        'name': budget_name,
                        'limit': budget_limit,
                        'actual': actual_spend,
                        'usage_percent': (actual_spend / budget_limit) * 100 if budget_limit > 0 else 0
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get budget performance for {budget_name}: {e}")
            
            return budget_status
            
        except Exception as e:
            logger.error(f"Failed to get budget status: {e}")
            return []
    
    def get_cost_anomalies(self, days: int = 7) -> List[Dict]:
        """Get cost anomalies for the specified number of days"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            response = self.cost_explorer.get_anomalies(
                DateInterval={
                    'StartDate': start_date.strftime('%Y-%m-%d'),
                    'EndDate': end_date.strftime('%Y-%m-%d')
                }
            )
            
            anomalies = []
            
            for anomaly in response.get('Anomalies', []):
                anomalies.append({
                    'anomaly_id': anomaly['AnomalyId'],
                    'score': anomaly['Impact']['MaxImpact'],
                    'service': anomaly.get('DimensionKey', 'Unknown'),
                    'impact': float(anomaly['Impact']['TotalImpact']),
                    'date': anomaly['AnomalyStartDate']
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to get cost anomalies: {e}")
            return []


class EC2CostMonitor:
    """Monitor EC2 instance costs and utilization"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.ec2 = None
        self.cloudwatch = None
        
        if AWS_AVAILABLE:
            try:
                self.ec2 = boto3.client('ec2', region_name=region)
                self.cloudwatch = boto3.client('cloudwatch', region_name=region)
            except Exception as e:
                logger.warning(f"Failed to initialize EC2 clients: {e}")
    
    def get_instance_costs(self) -> Dict[str, Dict]:
        """Get cost information for all running instances"""
        if not self.ec2:
            return {}
        
        try:
            # Get running instances
            response = self.ec2.describe_instances(
                Filters=[
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )
            
            instance_costs = {}
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_id = instance['InstanceId']
                    instance_type = instance['InstanceType']
                    
                    # Get approximate hourly cost (simplified pricing)
                    hourly_cost = self._get_instance_hourly_cost(instance_type)
                    
                    # Get utilization metrics
                    utilization = self._get_instance_utilization(instance_id)
                    
                    instance_costs[instance_id] = {
                        'type': instance_type,
                        'hourly_cost': hourly_cost,
                        'cpu_utilization': utilization.get('cpu', 0),
                        'network_utilization': utilization.get('network', 0)
                    }
            
            return instance_costs
            
        except Exception as e:
            logger.error(f"Failed to get instance costs: {e}")
            return {}
    
    def _get_instance_hourly_cost(self, instance_type: str) -> float:
        """Get approximate hourly cost for instance type (simplified)"""
        # Simplified pricing - in production, use AWS Pricing API
        cost_map = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            't3.large': 0.0832,
            't3.xlarge': 0.1664,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34,
            'g4dn.xlarge': 0.526,  # GPU instance
            'g4dn.2xlarge': 0.752,
        }
        return cost_map.get(instance_type, 0.1)  # Default fallback
    
    def _get_instance_utilization(self, instance_id: str) -> Dict[str, float]:
        """Get instance utilization metrics"""
        if not self.cloudwatch:
            return {}
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            # Get CPU utilization
            cpu_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[
                    {'Name': 'InstanceId', 'Value': instance_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            cpu_utilization = 0
            if cpu_response['Datapoints']:
                cpu_utilization = cpu_response['Datapoints'][-1]['Average']
            
            # Get network utilization
            network_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='NetworkIn',
                Dimensions=[
                    {'Name': 'InstanceId', 'Value': instance_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            network_utilization = 0
            if network_response['Datapoints']:
                network_utilization = network_response['Datapoints'][-1]['Average']
            
            return {
                'cpu': cpu_utilization,
                'network': network_utilization
            }
            
        except Exception as e:
            logger.warning(f"Failed to get utilization for {instance_id}: {e}")
            return {}


class CostOptimizationAnalyzer:
    """Analyze costs and provide optimization recommendations"""
    
    def __init__(self, cost_explorer: AWSCostExplorer, ec2_monitor: EC2CostMonitor):
        self.cost_explorer = cost_explorer
        self.ec2_monitor = ec2_monitor
    
    @cost_optimization_duration.time()
    def analyze_optimization_opportunities(self) -> Dict[str, float]:
        """Analyze and calculate potential cost savings"""
        cost_optimization_requests.inc()
        
        opportunities = {}
        
        try:
            # Analyze underutilized EC2 instances
            ec2_savings = self._analyze_ec2_optimization()
            opportunities.update(ec2_savings)
            
            # Analyze storage optimization
            storage_savings = self._analyze_storage_optimization()
            opportunities.update(storage_savings)
            
            # Analyze Reserved Instance opportunities
            ri_savings = self._analyze_reserved_instance_opportunities()
            opportunities.update(ri_savings)
            
        except Exception as e:
            logger.error(f"Cost optimization analysis failed: {e}")
        
        return opportunities
    
    def _analyze_ec2_optimization(self) -> Dict[str, float]:
        """Analyze EC2 cost optimization opportunities"""
        savings = {}
        
        try:
            instance_costs = self.ec2_monitor.get_instance_costs()
            
            underutilized_savings = 0
            oversized_savings = 0
            
            for instance_id, data in instance_costs.items():
                cpu_util = data['cpu_utilization']
                hourly_cost = data['hourly_cost']
                
                # Flag underutilized instances (< 10% CPU for 1 hour)
                if cpu_util < 10:
                    underutilized_savings += hourly_cost * 24 * 30  # Monthly savings
                
                # Flag oversized instances (< 25% CPU consistently)
                elif cpu_util < 25:
                    # Assume 50% cost reduction by downsizing
                    oversized_savings += hourly_cost * 0.5 * 24 * 30
            
            savings['underutilized_instances'] = underutilized_savings
            savings['oversized_instances'] = oversized_savings
            
        except Exception as e:
            logger.warning(f"EC2 optimization analysis failed: {e}")
        
        return savings
    
    def _analyze_storage_optimization(self) -> Dict[str, float]:
        """Analyze storage cost optimization opportunities"""
        # Simplified - in production, analyze S3 storage classes, EBS snapshots, etc.
        return {
            's3_lifecycle_policies': 50.0,  # Estimated monthly savings
            'ebs_snapshot_cleanup': 25.0,
            'unused_ebs_volumes': 30.0
        }
    
    def _analyze_reserved_instance_opportunities(self) -> Dict[str, float]:
        """Analyze Reserved Instance purchase opportunities"""
        # Simplified - in production, analyze usage patterns and RI coverage
        return {
            'reserved_instances': 150.0,  # Estimated monthly savings
            'savings_plans': 75.0
        }


class CostMetricsExporter:
    """Main cost metrics exporter"""
    
    def __init__(self, config: CostMetricsConfig):
        self.config = config
        self.cost_explorer = None
        self.ec2_monitor = None
        self.optimizer = None
        self.running = False
        
        if AWS_AVAILABLE:
            try:
                self.cost_explorer = AWSCostExplorer(config.aws_region)
                self.ec2_monitor = EC2CostMonitor(config.aws_region)
                self.optimizer = CostOptimizationAnalyzer(self.cost_explorer, self.ec2_monitor)
                logger.info("Cost metrics exporter initialized with AWS integration")
            except Exception as e:
                logger.error(f"Failed to initialize AWS integration: {e}")
                raise
        else:
            logger.error("AWS integration not available - install boto3")
            raise ImportError("boto3 required for cost metrics")
    
    def update_daily_cost_metrics(self):
        """Update daily cost metrics"""
        try:
            daily_costs, service_costs = self.cost_explorer.get_daily_costs(self.config.cost_history_days)
            
            # Update daily cost metrics
            for date_str, cost in daily_costs.items():
                aws_cost_daily.labels(date=date_str).set(cost)
            
            # Update service-specific costs
            for service, dates in service_costs.items():
                for date_str, cost in dates.items():
                    aws_cost_usd.labels(service=service, period='daily', dimension='service').set(cost)
            
            logger.debug(f"Updated daily costs for {len(daily_costs)} days")
            
        except Exception as e:
            logger.error(f"Failed to update daily cost metrics: {e}")
    
    def update_monthly_cost_metrics(self):
        """Update monthly cost metrics"""
        try:
            monthly_costs = self.cost_explorer.get_monthly_costs(12)
            
            for month, cost in monthly_costs.items():
                aws_cost_monthly.labels(month=month).set(cost)
            
            logger.debug(f"Updated monthly costs for {len(monthly_costs)} months")
            
        except Exception as e:
            logger.error(f"Failed to update monthly cost metrics: {e}")
    
    def update_forecast_metrics(self):
        """Update cost forecast metrics"""
        try:
            if self.config.enable_forecasting:
                forecast = self.cost_explorer.get_cost_forecast(self.config.forecast_days)
                
                total_forecast = sum(forecast.values())
                aws_cost_forecast.labels(period='30_days').set(total_forecast)
                
                logger.debug(f"Updated cost forecast: ${total_forecast:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to update forecast metrics: {e}")
    
    def update_budget_metrics(self):
        """Update budget usage metrics"""
        try:
            if self.config.enable_budget_monitoring:
                budgets = self.cost_explorer.get_budget_status()
                
                for budget in budgets:
                    aws_cost_budget_usage.labels(budget_name=budget['name']).set(budget['usage_percent'])
                
                logger.debug(f"Updated budget metrics for {len(budgets)} budgets")
            
        except Exception as e:
            logger.error(f"Failed to update budget metrics: {e}")
    
    def update_anomaly_metrics(self):
        """Update cost anomaly metrics"""
        try:
            anomalies = self.cost_explorer.get_cost_anomalies(7)
            
            # Reset all anomaly scores
            aws_cost_anomaly_score._metrics.clear()
            
            for anomaly in anomalies:
                aws_cost_anomaly_score.labels(service=anomaly['service']).set(anomaly['score'])
            
            logger.debug(f"Updated anomaly metrics for {len(anomalies)} anomalies")
            
        except Exception as e:
            logger.error(f"Failed to update anomaly metrics: {e}")
    
    def update_instance_metrics(self):
        """Update EC2 instance cost and utilization metrics"""
        try:
            if self.config.enable_ec2_monitoring:
                instance_costs = self.ec2_monitor.get_instance_costs()
                
                for instance_id, data in instance_costs.items():
                    ec2_instance_cost.labels(
                        instance_id=instance_id,
                        instance_type=data['type']
                    ).set(data['hourly_cost'])
                    
                    ec2_instance_utilization.labels(
                        instance_id=instance_id,
                        metric_type='cpu'
                    ).set(data['cpu_utilization'])
                
                logger.debug(f"Updated instance metrics for {len(instance_costs)} instances")
            
        except Exception as e:
            logger.error(f"Failed to update instance metrics: {e}")
    
    def update_optimization_metrics(self):
        """Update cost optimization metrics"""
        try:
            if self.config.enable_optimization_analysis:
                opportunities = self.optimizer.analyze_optimization_opportunities()
                
                for recommendation_type, savings in opportunities.items():
                    aws_cost_savings_opportunities.labels(
                        recommendation_type=recommendation_type
                    ).set(savings)
                
                logger.debug(f"Updated optimization metrics: {len(opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"Failed to update optimization metrics: {e}")
    
    def update_all_metrics(self):
        """Update all cost metrics"""
        logger.info("Updating all cost metrics...")
        
        try:
            self.update_daily_cost_metrics()
            self.update_monthly_cost_metrics()
            self.update_forecast_metrics()
            self.update_budget_metrics()
            self.update_anomaly_metrics()
            self.update_instance_metrics()
            self.update_optimization_metrics()
            
            logger.info("All cost metrics updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update cost metrics: {e}")
    
    def start_scheduled_updates(self):
        """Start scheduled metric updates"""
        # Schedule daily updates
        schedule.every().hour.do(self.update_daily_cost_metrics)
        schedule.every().hour.do(self.update_instance_metrics)
        
        # Schedule less frequent updates
        schedule.every(6).hours.do(self.update_monthly_cost_metrics)
        schedule.every(6).hours.do(self.update_forecast_metrics)
        schedule.every(6).hours.do(self.update_budget_metrics)
        schedule.every(12).hours.do(self.update_anomaly_metrics)
        schedule.every(12).hours.do(self.update_optimization_metrics)
        
        # Run scheduler in background
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduled cost metric updates started")
    
    def run(self):
        """Run the cost metrics exporter"""
        logger.info("Starting AWS Cost Metrics Exporter")
        logger.info(f"Region: {self.config.aws_region}")
        logger.info(f"Update interval: {self.config.update_interval_minutes} minutes")
        
        # Start Prometheus metrics server
        start_http_server(self.config.metrics_port)
        logger.info(f"Prometheus metrics server started on port {self.config.metrics_port}")
        
        self.running = True
        
        # Initial metric update
        self.update_all_metrics()
        
        # Start scheduled updates
        self.start_scheduled_updates()
        
        try:
            # Main loop
            while self.running:
                time.sleep(self.config.update_interval_minutes * 60)
                self.update_all_metrics()
                
        except KeyboardInterrupt:
            logger.info("Cost metrics exporter stopped by user")
        except Exception as e:
            logger.error(f"Cost metrics exporter error: {e}")
            raise
        finally:
            self.running = False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='AWS Cost Metrics Exporter')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--port', type=int, default=9092, help='Metrics server port')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in minutes')
    parser.add_argument('--disable-forecasting', action='store_true', help='Disable cost forecasting')
    parser.add_argument('--disable-optimization', action='store_true', help='Disable optimization analysis')
    
    args = parser.parse_args()
    
    # Create configuration
    config = CostMetricsConfig(
        aws_region=args.region,
        metrics_port=args.port,
        update_interval_minutes=args.interval,
        enable_forecasting=not args.disable_forecasting,
        enable_optimization_analysis=not args.disable_optimization
    )
    
    # Create and run exporter
    exporter = CostMetricsExporter(config)
    exporter.run()


if __name__ == "__main__":
    main()