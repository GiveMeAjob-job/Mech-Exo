#!/usr/bin/env python3
"""
Risk Metrics Prometheus Exporter

Exports /riskz API metrics to Prometheus format for monitoring and alerting.
Converts risk dashboard data into Prometheus gauges with proper labels.

Usage:
    python prometheus/risk_exporter.py --port 8000 --risk-url http://localhost:8050
    
Metrics exposed at: http://localhost:8000/metrics
"""

import os
import sys
import time
import json
import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from prometheus_client import start_http_server, Gauge, Counter, Info, CollectorRegistry
    from prometheus_client.core import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("❌ prometheus_client not available. Install with: pip install prometheus_client")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    var_95: float
    var_99: float
    today_pnl_pct: float
    month_pnl_pct: float
    position_count: int
    total_exposure: float
    ops_ok: int  # 1 = operational, 0 = down
    last_updated: str


class RiskExporter:
    """Prometheus exporter for risk metrics"""
    
    def __init__(self, risk_url: str = "http://localhost:8050", scrape_interval: int = 30, environment: str = "prod"):
        self.risk_url = risk_url.rstrip('/')
        self.scrape_interval = scrape_interval
        self.environment = environment
        
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # Define Prometheus metrics
        self._define_metrics()
        
        # Track export statistics
        self.last_scrape_time = 0
        self.scrape_count = 0
        self.error_count = 0
        
        logger.info(f"🔧 Risk Exporter initialized")
        logger.info(f"   Risk URL: {self.risk_url}")
        logger.info(f"   Scrape Interval: {scrape_interval}s")
        
    def _define_metrics(self):
        """Define all Prometheus metrics"""
        
        # Core Risk Metrics
        self.var_95_gauge = Gauge(
            'risk_var_95_usd',
            'Value at Risk 95% confidence in USD',
            registry=self.registry
        )
        
        self.var_99_gauge = Gauge(
            'risk_var_99_usd', 
            'Value at Risk 99% confidence in USD',
            registry=self.registry
        )
        
        self.today_pnl_gauge = Gauge(
            'risk_today_pnl_pct',
            'Today profit/loss percentage',
            registry=self.registry
        )
        
        self.month_pnl_gauge = Gauge(
            'risk_month_pnl_pct',
            'Month-to-date profit/loss percentage', 
            registry=self.registry
        )
        
        # Position Metrics
        self.position_count_gauge = Gauge(
            'risk_position_count',
            'Total number of positions',
            registry=self.registry
        )
        
        self.total_exposure_gauge = Gauge(
            'risk_total_exposure_usd',
            'Total portfolio exposure in USD',
            registry=self.registry
        )
        
        # System Health
        self.ops_ok_gauge = Gauge(
            'risk_ops_ok',
            'System operational status (1=OK, 0=Down)',
            ['env'],  # Add environment label for SLO tracking
            registry=self.registry
        )
        
        self.data_age_gauge = Gauge(
            'risk_data_age_seconds',
            'Age of risk data in seconds',
            registry=self.registry
        )
        
        # VaR Utilization (percentage of limit)
        self.var_95_utilization_gauge = Gauge(
            'risk_var_95_utilization_pct',
            'VaR 95% utilization as percentage of limit',
            registry=self.registry
        )
        
        self.var_99_utilization_gauge = Gauge(
            'risk_var_99_utilization_pct',
            'VaR 99% utilization as percentage of limit', 
            registry=self.registry
        )
        
        # Alert Levels
        self.alert_level_gauge = Gauge(
            'risk_alert_level',
            'Current alert level (0=normal, 1=warning, 2=critical)',
            registry=self.registry
        )
        
        # Exporter Statistics  
        self.scrape_duration_gauge = Gauge(
            'risk_exporter_scrape_duration_seconds',
            'Time spent scraping risk API',
            registry=self.registry
        )
        
        self.scrape_counter = Counter(
            'risk_exporter_scrapes_total',
            'Total number of scrapes attempted',
            ['status'],  # success, error
            registry=self.registry
        )
        
        # System Info
        self.info_gauge = Info(
            'risk_exporter_info',
            'Risk exporter version and config',
            registry=self.registry
        )
        
        # Set static info
        self.info_gauge.info({
            'version': '1.0.0',
            'risk_url': self.risk_url,
            'scrape_interval': str(self.scrape_interval)
        })
        
    def fetch_risk_data(self) -> Optional[RiskMetrics]:
        """Fetch risk data from /riskz endpoint"""
        start_time = time.time()
        
        try:
            # Try /riskz endpoint first
            response = requests.get(
                f"{self.risk_url}/riskz",
                timeout=10
            )
            
            scrape_duration = time.time() - start_time
            self.scrape_duration_gauge.set(scrape_duration)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract metrics from response
                metrics = RiskMetrics(
                    var_95=float(data.get('var_95', 0)),
                    var_99=float(data.get('var_99', 0)),
                    today_pnl_pct=float(data.get('today_pnl_pct', 0)),
                    month_pnl_pct=float(data.get('month_pnl_pct', 0)),
                    position_count=int(data.get('position_count', 0)),
                    total_exposure=float(data.get('total_exposure', 0)),
                    ops_ok=1,  # If we got data, system is operational
                    last_updated=data.get('last_updated', datetime.now().isoformat())
                )
                
                self.scrape_counter.labels(status='success').inc()
                return metrics
                
            else:
                logger.warning(f"Risk API returned {response.status_code}")
                self.scrape_counter.labels(status='error').inc()
                return None
                
        except requests.exceptions.RequestException as e:
            scrape_duration = time.time() - start_time
            self.scrape_duration_gauge.set(scrape_duration)
            
            logger.error(f"Failed to fetch risk data: {str(e)}")
            self.scrape_counter.labels(status='error').inc()
            return None
            
    def calculate_derived_metrics(self, metrics: RiskMetrics) -> Dict[str, float]:
        """Calculate derived metrics from raw data"""
        
        # VaR limits (configurable)
        var_95_limit = float(os.getenv('RISK_VAR_95_LIMIT', '2000000'))  # $2M
        var_99_limit = float(os.getenv('RISK_VAR_99_LIMIT', '3000000'))  # $3M
        
        # Calculate utilization percentages
        var_95_util = (metrics.var_95 / var_95_limit * 100) if var_95_limit > 0 else 0
        var_99_util = (metrics.var_99 / var_99_limit * 100) if var_99_limit > 0 else 0
        
        # Calculate alert level
        alert_level = 0  # Normal
        if var_95_util > 90 or metrics.today_pnl_pct < -0.8 or metrics.month_pnl_pct < -2.5:
            alert_level = 2  # Critical
        elif var_95_util > 70 or metrics.today_pnl_pct < -0.4 or metrics.month_pnl_pct < -2.0:
            alert_level = 1  # Warning
            
        # Calculate data age
        try:
            last_updated = datetime.fromisoformat(metrics.last_updated.replace('Z', '+00:00'))
            data_age = (datetime.now() - last_updated.replace(tzinfo=None)).total_seconds()
        except:
            data_age = 0
            
        return {
            'var_95_utilization': var_95_util,
            'var_99_utilization': var_99_util,
            'alert_level': alert_level,
            'data_age': data_age
        }
        
    def update_metrics(self, metrics: RiskMetrics):
        """Update all Prometheus metrics"""
        
        # Core metrics
        self.var_95_gauge.set(metrics.var_95)
        self.var_99_gauge.set(metrics.var_99)
        self.today_pnl_gauge.set(metrics.today_pnl_pct)
        self.month_pnl_gauge.set(metrics.month_pnl_pct)
        self.position_count_gauge.set(metrics.position_count)
        self.total_exposure_gauge.set(metrics.total_exposure)
        self.ops_ok_gauge.labels(env=self.environment).set(metrics.ops_ok)
        
        # Derived metrics
        derived = self.calculate_derived_metrics(metrics)
        self.var_95_utilization_gauge.set(derived['var_95_utilization'])
        self.var_99_utilization_gauge.set(derived['var_99_utilization'])
        self.alert_level_gauge.set(derived['alert_level'])
        self.data_age_gauge.set(derived['data_age'])
        
        logger.debug(f"Updated metrics: VaR95=${metrics.var_95:.0f}, PnL={metrics.today_pnl_pct:+.2f}%")
        
    def run_scrape_loop(self):
        """Main scrape loop"""
        logger.info("🚀 Starting risk metrics scrape loop")
        
        while True:
            try:
                metrics = self.fetch_risk_data()
                
                if metrics:
                    self.update_metrics(metrics)
                    self.scrape_count += 1
                    
                    if self.scrape_count % 10 == 0:  # Log every 10 scrapes
                        logger.info(f"✅ Scraped {self.scrape_count} times, "
                                  f"VaR95=${metrics.var_95:.0f}, "
                                  f"PnL={metrics.today_pnl_pct:+.2f}%")
                else:
                    # Set ops_ok to 0 if we can't get data
                    self.ops_ok_gauge.labels(env=self.environment).set(0)
                    self.error_count += 1
                    
                    if self.error_count % 5 == 0:  # Log every 5 errors
                        logger.warning(f"❌ Failed to scrape {self.error_count} times")
                
                self.last_scrape_time = time.time()
                time.sleep(self.scrape_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Scrape loop interrupted")
                break
            except Exception as e:
                logger.error(f"Scrape loop error: {str(e)}")
                time.sleep(self.scrape_interval)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Risk Metrics Prometheus Exporter')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to serve metrics on (default: 8000)')
    parser.add_argument('--risk-url', default='http://localhost:8050',
                       help='Risk API URL (default: http://localhost:8050)')
    parser.add_argument('--scrape-interval', type=int, default=30,
                       help='Scrape interval in seconds (default: 30)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create exporter
    exporter = RiskExporter(
        risk_url=args.risk_url,
        scrape_interval=args.scrape_interval
    )
    
    # Start HTTP server for metrics
    logger.info(f"🌐 Starting Prometheus metrics server on port {args.port}")
    start_http_server(args.port, registry=exporter.registry)
    
    logger.info(f"📊 Metrics available at: http://localhost:{args.port}/metrics")
    
    # Start scraping
    try:
        exporter.run_scrape_loop()
    except KeyboardInterrupt:
        logger.info("👋 Exporter shutdown")


if __name__ == '__main__':
    main()