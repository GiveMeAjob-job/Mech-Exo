#!/usr/bin/env python3
"""
Load Runner for Mech-Exo Risk Control System
Generates concurrent order placement calls for load testing
"""

import time
import json
import logging
import random
import argparse
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadConfig:
    """Load test configuration"""
    target_rate: int = 50  # req/sec
    ramp_duration: int = 60  # seconds
    test_duration: int = 3600  # seconds
    max_concurrent: int = 100
    base_url: str = "http://localhost:8050"
    paper_trading: bool = True
    error_threshold: float = 0.03  # 3% error rate threshold


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    order_type: str = 'market'
    paper_trade: bool = True


@dataclass
class TestMetrics:
    """Test execution metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    error_rate: float = 0.0
    current_rate: float = 0.0
    start_time: Optional[datetime] = None
    
    def update_request(self, latency: float, success: bool):
        """Update metrics with request result"""
        self.total_requests += 1
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        self.error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total_requests if self.total_requests > 0 else 0.0


class LoadGenerator:
    """Load test generator"""
    
    def __init__(self, config: LoadConfig):
        self.config = config
        self.metrics = TestMetrics()
        self.session = requests.Session()
        self.running = False
        self.start_time = None
        self.metrics_lock = threading.Lock()
        
        # Configure session
        self.session.timeout = 30
        
        # Order templates for testing
        self.order_templates = [
            OrderRequest("AAPL", "buy", 100, paper_trade=True),
            OrderRequest("GOOGL", "sell", 50, paper_trade=True),
            OrderRequest("MSFT", "buy", 75, paper_trade=True),
            OrderRequest("TSLA", "sell", 25, paper_trade=True),
            OrderRequest("NVDA", "buy", 40, paper_trade=True),
        ]
    
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            self.session.close()
    
    def generate_order(self) -> OrderRequest:
        """Generate a random order for testing"""
        template = random.choice(self.order_templates)
        
        # Add some randomization
        quantity_multiplier = random.uniform(0.5, 2.0)
        order = OrderRequest(
            symbol=template.symbol,
            side=template.side,
            quantity=int(template.quantity * quantity_multiplier),
            order_type='market',
            paper_trade=self.config.paper_trading
        )
        
        return order
    
    def send_order(self, order: OrderRequest) -> Tuple[bool, float]:
        """Send a single order request"""
        start_time = time.time()
        
        try:
            # Convert order to API payload
            payload = {
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'type': order.order_type,
                'paper_trade': order.paper_trade
            }
            
            # Send to risk API order endpoint  
            response = self.session.post(
                f"{self.config.base_url}/api/orders",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                return True, latency
            else:
                logger.warning(f"Order failed: {response.status_code} - {response.text}")
                return False, latency
                
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Order request exception: {str(e)}")
            return False, latency
    
    def health_check(self) -> bool:
        """Check if the target system is healthy"""
        try:
            response = self.session.get(f"{self.config.base_url}/healthz", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def calculate_target_rate(self, elapsed_time: float) -> float:
        """Calculate current target rate based on ramp-up"""
        if elapsed_time < self.config.ramp_duration:
            # Ramp up linearly
            ramp_progress = elapsed_time / self.config.ramp_duration
            return self.config.target_rate * ramp_progress
        else:
            # Steady state
            return self.config.target_rate
    
    def worker_task(self, worker_id: int):
        """Worker thread for sending requests"""
        logger.info(f"Worker {worker_id} started")
        last_request_time = time.time()
        
        while self.running:
            elapsed_time = time.time() - self.start_time
            
            # Check if test duration exceeded
            if elapsed_time > self.config.test_duration:
                logger.info(f"Worker {worker_id} stopping - test duration exceeded")
                break
            
            # Calculate current target rate per worker
            total_target_rate = self.calculate_target_rate(elapsed_time)
            num_workers = min(self.config.max_concurrent, max(1, int(self.config.target_rate / 10)))
            worker_rate = total_target_rate / num_workers
            
            # Check error rate threshold
            with self.metrics_lock:
                if self.metrics.error_rate > self.config.error_threshold:
                    logger.error(f"Error rate {self.metrics.error_rate:.2%} exceeds threshold {self.config.error_threshold:.2%}")
                    self.running = False
                    break
            
            # Rate limiting
            if worker_rate > 0:
                interval = 1.0 / worker_rate
                time_since_last = time.time() - last_request_time
                if time_since_last < interval:
                    time.sleep(interval - time_since_last)
            
            # Generate and send order
            order = self.generate_order()
            success, latency = self.send_order(order)
            
            # Update metrics (thread-safe)
            with self.metrics_lock:
                self.metrics.update_request(latency, success)
                self.metrics.current_rate = total_target_rate
            
            last_request_time = time.time()
        
        logger.info(f"Worker {worker_id} finished")
    
    def monitor_task(self):
        """Monitor and log test progress"""
        logger.info("Monitor started")
        last_log_time = time.time()
        
        while self.running:
            time.sleep(10)  # Log every 10 seconds
            
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            if current_time - last_log_time >= 10:  # Log every 10 seconds
                with self.metrics_lock:
                    logger.info(
                        f"Progress: {elapsed_time:.0f}s | "
                        f"Rate: {self.metrics.current_rate:.1f} req/s | "
                        f"Total: {self.metrics.total_requests} | "
                        f"Success: {self.metrics.successful_requests} | "
                        f"Errors: {self.metrics.failed_requests} ({self.metrics.error_rate:.2%}) | "
                        f"Avg Latency: {self.metrics.avg_latency*1000:.1f}ms"
                    )
                last_log_time = current_time
                
                # Export metrics to Prometheus format
                self.export_metrics()
        
        logger.info("Monitor finished")
    
    def export_metrics(self):
        """Export metrics in Prometheus format"""
        try:
            with self.metrics_lock:
                metrics_data = {
                    'order_req_rate': self.metrics.current_rate,
                    'order_err_rate': self.metrics.error_rate * 100,  # Convert to percentage
                    'latency_ms': self.metrics.avg_latency * 1000,  # Convert to milliseconds
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests
                }
            
            # Write metrics to file for Prometheus scraping
            metrics_file = Path("load_test_metrics.prom")
            with open(metrics_file, 'w') as f:
                for metric_name, value in metrics_data.items():
                    f.write(f"# HELP {metric_name} Load test metric\n")
                    f.write(f"# TYPE {metric_name} gauge\n")
                    f.write(f"{metric_name} {value}\n")
                    
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
    
    def run_test(self) -> TestMetrics:
        """Run the complete load test"""
        logger.info("Starting load test")
        logger.info(f"Config: {self.config.target_rate} req/s for {self.config.test_duration}s")
        
        # Initial health check
        if not self.health_check():
            logger.error("Initial health check failed - continuing anyway for testing")
        
        self.running = True
        self.start_time = time.time()
        self.metrics.start_time = datetime.now()
        
        # Calculate number of workers
        num_workers = min(self.config.max_concurrent, max(1, int(self.config.target_rate / 10)))
        
        # Start workers and monitor
        threads = []
        
        # Start worker threads
        for i in range(num_workers):
            thread = threading.Thread(target=self.worker_task, args=(i,), daemon=True)
            thread.start()
            threads.append(thread)
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=self.monitor_task, daemon=True)
        monitor_thread.start()
        threads.append(monitor_thread)
        
        try:
            # Wait for test duration
            time.sleep(self.config.test_duration)
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            # Clean shutdown
            self.running = False
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=10)
        
        # Generate final report
        test_duration = time.time() - self.start_time
        logger.info("Load test completed")
        logger.info(f"Duration: {test_duration:.1f}s")
        logger.info(f"Total requests: {self.metrics.total_requests}")
        logger.info(f"Successful requests: {self.metrics.successful_requests}")
        logger.info(f"Failed requests: {self.metrics.failed_requests}")
        logger.info(f"Error rate: {self.metrics.error_rate:.2%}")
        logger.info(f"Average latency: {self.metrics.avg_latency*1000:.1f}ms")
        logger.info(f"Min latency: {self.metrics.min_latency*1000:.1f}ms")
        logger.info(f"Max latency: {self.metrics.max_latency*1000:.1f}ms")
        
        return self.metrics


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Load test runner for Mech-Exo')
    parser.add_argument('--rate', type=int, default=50,
                       help='Target request rate (req/sec)')
    parser.add_argument('--ramp', type=int, default=60,
                       help='Ramp-up duration (seconds)')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Test duration (seconds)')
    parser.add_argument('--url', default='http://localhost:8050',
                       help='Base URL for the API')
    parser.add_argument('--paper', action='store_true', default=True,
                       help='Use paper trading mode')
    parser.add_argument('--config', type=str,
                       help='Load test configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = LoadConfig(
        target_rate=args.rate,
        ramp_duration=args.ramp,
        test_duration=args.duration,
        base_url=args.url,
        paper_trading=args.paper
    )
    
    # Override with config file if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
    
    # Run the load test
    with LoadGenerator(config) as generator:
        try:
            metrics = generator.run_test()
            
            # Save results
            results = {
                'config': asdict(config),
                'metrics': asdict(metrics),
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            # Return appropriate exit code
            if metrics.error_rate > config.error_threshold:
                logger.error("Load test failed - error rate too high")
                return 1
            else:
                logger.info("Load test passed")
                return 0
                
        except Exception as e:
            logger.error(f"Load test failed with exception: {str(e)}")
            return 1


if __name__ == "__main__":
    sys.exit(main())