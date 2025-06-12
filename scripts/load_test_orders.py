#!/usr/bin/env python3
"""
Load Testing Script - 10x Volume Stress Test

Simulates high-volume order flow to test system performance under stress.
Measures order routing latency, database write performance, and system stability.
"""

import argparse
import concurrent.futures
import json
import logging
import random
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderLoadTester:
    """Manages high-volume order load testing"""
    
    def __init__(self, db_path: str = "data/load_test.duckdb"):
        """
        Initialize load tester
        
        Args:
            db_path: Path to test database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
        self.order_types = ['market', 'limit', 'stop']
        self.strategies = ['momentum', 'mean_reversion', 'ml_alpha', 'arbitrage']
        
        # Performance tracking
        self.latency_measurements = []
        self.error_count = 0
        self.total_orders = 0
        
        # Initialize test database
        self.setup_database()
    
    def setup_database(self):
        """Setup test database with required tables"""
        try:
            from mech_exo.datasource.storage import DataStorage
            
            storage = DataStorage(str(self.db_path))
            
            # Create orders table for load testing
            storage.conn.execute("""
                CREATE TABLE IF NOT EXISTS load_test_orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    price REAL,
                    strategy TEXT,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms REAL,
                    status TEXT DEFAULT 'pending'
                )
            """)
            
            # Create fills table for load testing
            storage.conn.execute("""
                CREATE TABLE IF NOT EXISTS load_test_fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    filled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms REAL,
                    FOREIGN KEY (order_id) REFERENCES load_test_orders (order_id)
                )
            """)
            
            # Create performance metrics table
            storage.conn.execute("""
                CREATE TABLE IF NOT EXISTS load_test_metrics (
                    test_run_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    metric_value REAL,
                    details TEXT
                )
            """)
            
            storage.conn.commit()
            storage.close()
            
            logger.info(f"✅ Load test database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup database: {e}")
            raise
    
    def generate_mock_order(self) -> Dict[str, Any]:
        """Generate a realistic mock order"""
        symbol = random.choice(self.test_symbols)
        order_type = random.choice(self.order_types)
        strategy = random.choice(self.strategies)
        
        # Generate realistic quantities (100-10000 shares)
        quantity = random.randint(1, 100) * 100
        
        # Randomly choose buy/sell
        if random.random() > 0.5:
            quantity = -quantity  # Sell order
        
        # Generate realistic prices
        base_prices = {
            'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 420.0, 'AMZN': 3400.0,
            'TSLA': 250.0, 'META': 500.0, 'NVDA': 800.0, 'NFLX': 600.0,
            'AMD': 120.0, 'INTC': 60.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add some random variation (±5%)
        price_variation = random.uniform(0.95, 1.05)
        price = round(base_price * price_variation, 2)
        
        # For limit orders, adjust price slightly
        if order_type == 'limit':
            if quantity > 0:  # Buy order - bid lower
                price *= random.uniform(0.98, 0.995)
            else:  # Sell order - ask higher
                price *= random.uniform(1.005, 1.02)
        
        return {
            'order_id': str(uuid.uuid4()),
            'symbol': symbol,
            'quantity': quantity,
            'order_type': order_type,
            'price': round(price, 2),
            'strategy': strategy,
            'timestamp': datetime.now()
        }
    
    def process_single_order(self, order_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Process a single order and measure latency
        
        Args:
            order_data: Order details
            
        Returns:
            Tuple of (success, latency_ms)
        """
        start_time = time.perf_counter()
        
        try:
            from mech_exo.datasource.storage import DataStorage
            
            # Simulate order processing
            storage = DataStorage(str(self.db_path))
            
            # Insert order into database
            storage.conn.execute("""
                INSERT INTO load_test_orders 
                (order_id, symbol, quantity, order_type, price, strategy, processing_time_ms, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                order_data['order_id'],
                order_data['symbol'],
                order_data['quantity'],
                order_data['order_type'],
                order_data['price'],
                order_data['strategy'],
                0,  # Will be updated below
                'submitted'
            ])
            
            # Simulate order routing and validation logic
            self.simulate_order_routing(order_data)
            
            # Simulate order execution (create fill)
            if random.random() > 0.05:  # 95% fill rate
                fill_price = self.simulate_fill_price(order_data)
                
                storage.conn.execute("""
                    INSERT INTO load_test_fills 
                    (fill_id, order_id, symbol, quantity, price, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    str(uuid.uuid4()),
                    order_data['order_id'],
                    order_data['symbol'],
                    order_data['quantity'],
                    fill_price,
                    0  # Will be updated below
                ])
                
                # Update order status
                storage.conn.execute("""
                    UPDATE load_test_orders 
                    SET status = 'filled' 
                    WHERE order_id = ?
                """, [order_data['order_id']])
            else:
                # Order rejected
                storage.conn.execute("""
                    UPDATE load_test_orders 
                    SET status = 'rejected' 
                    WHERE order_id = ?
                """, [order_data['order_id']])
            
            storage.conn.commit()
            storage.close()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Update processing time in database
            storage = DataStorage(str(self.db_path))
            storage.conn.execute("""
                UPDATE load_test_orders 
                SET processing_time_ms = ? 
                WHERE order_id = ?
            """, [latency_ms, order_data['order_id']])
            storage.conn.commit()
            storage.close()
            
            return True, latency_ms
            
        except Exception as e:
            logger.error(f"Order processing failed: {e}")
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            return False, latency_ms
    
    def simulate_order_routing(self, order_data: Dict[str, Any]):
        """Simulate order routing logic with realistic delay"""
        # Simulate routing decision time
        time.sleep(random.uniform(0.001, 0.005))  # 1-5ms
        
        # Simulate risk checks
        time.sleep(random.uniform(0.002, 0.008))  # 2-8ms
    
    def simulate_fill_price(self, order_data: Dict[str, Any]) -> float:
        """Simulate realistic fill price with slippage"""
        base_price = order_data['price']
        
        if order_data['order_type'] == 'market':
            # Market orders have more slippage
            slippage = random.uniform(0.001, 0.003)  # 0.1-0.3%
            if order_data['quantity'] > 0:  # Buy
                return round(base_price * (1 + slippage), 2)
            else:  # Sell
                return round(base_price * (1 - slippage), 2)
        else:
            # Limit orders filled at limit price or better
            improvement = random.uniform(0, 0.001)  # 0-0.1% price improvement
            if order_data['quantity'] > 0:  # Buy
                return round(base_price * (1 - improvement), 2)
            else:  # Sell
                return round(base_price * (1 + improvement), 2)
    
    def run_load_test(self, num_orders: int = 5000, max_workers: int = 50) -> Dict[str, Any]:
        """
        Run the load test with specified parameters
        
        Args:
            num_orders: Number of orders to generate (default 5000 = 10x normal volume)
            max_workers: Maximum concurrent workers
            
        Returns:
            Test results dictionary
        """
        logger.info(f"🚀 Starting load test with {num_orders} orders, {max_workers} workers")
        
        # Generate all orders upfront
        logger.info("📋 Generating mock orders...")
        orders = [self.generate_mock_order() for _ in range(num_orders)]
        
        # Track test run
        test_run_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        # Process orders concurrently
        logger.info("⚡ Processing orders with concurrent workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all orders
            future_to_order = {
                executor.submit(self.process_single_order, order): order 
                for order in orders
            }
            
            # Collect results
            completed = 0
            for future in concurrent.futures.as_completed(future_to_order):
                order = future_to_order[future]
                try:
                    success, latency_ms = future.result()
                    self.latency_measurements.append(latency_ms)
                    self.total_orders += 1
                    
                    if not success:
                        self.error_count += 1
                    
                    completed += 1
                    if completed % 500 == 0:
                        logger.info(f"Processed {completed}/{num_orders} orders...")
                        
                except Exception as e:
                    logger.error(f"Order failed: {e}")
                    self.error_count += 1
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        # Calculate statistics
        results = self.calculate_results(test_run_id, total_duration, num_orders)
        
        # Save results to database
        self.save_results(test_run_id, results)
        
        return results
    
    def calculate_results(self, test_run_id: str, total_duration: float, num_orders: int) -> Dict[str, Any]:
        """Calculate test results and statistics"""
        
        if not self.latency_measurements:
            return {'error': 'No successful orders processed'}
        
        # Latency statistics
        latencies = self.latency_measurements
        latency_stats = {
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'p95_ms': self.percentile(latencies, 95),
            'p99_ms': self.percentile(latencies, 99)
        }
        
        # Throughput statistics
        successful_orders = self.total_orders - self.error_count
        throughput_stats = {
            'total_orders': self.total_orders,
            'successful_orders': successful_orders,
            'failed_orders': self.error_count,
            'success_rate_pct': (successful_orders / self.total_orders * 100) if self.total_orders > 0 else 0,
            'orders_per_second': successful_orders / total_duration if total_duration > 0 else 0,
            'total_duration_sec': total_duration
        }
        
        # Performance assessment
        p95_latency = latency_stats['p95_ms']
        performance_grade = 'PASS' if p95_latency < 500 else 'FAIL'
        
        results = {
            'test_run_id': test_run_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'target_orders': num_orders,
                'actual_orders': self.total_orders
            },
            'latency': latency_stats,
            'throughput': throughput_stats,
            'performance': {
                'grade': performance_grade,
                'p95_target_ms': 500,
                'p95_actual_ms': p95_latency,
                'target_met': p95_latency < 500
            }
        }
        
        return results
    
    @staticmethod
    def percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def save_results(self, test_run_id: str, results: Dict[str, Any]):
        """Save results to database and generate report"""
        try:
            from mech_exo.datasource.storage import DataStorage
            
            storage = DataStorage(str(self.db_path))
            
            # Save metrics to database
            metrics = [
                ('latency_p95_ms', results['latency']['p95_ms']),
                ('latency_mean_ms', results['latency']['mean_ms']),
                ('throughput_ops', results['throughput']['orders_per_second']),
                ('success_rate_pct', results['throughput']['success_rate_pct']),
                ('total_duration_sec', results['throughput']['total_duration_sec'])
            ]
            
            for metric_name, metric_value in metrics:
                storage.conn.execute("""
                    INSERT INTO load_test_metrics (test_run_id, metric_name, metric_value, details)
                    VALUES (?, ?, ?, ?)
                """, [test_run_id, metric_name, metric_value, json.dumps(results)])
            
            storage.conn.commit()
            storage.close()
            
            # Generate markdown report
            self.generate_report(results)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate markdown report"""
        
        report_path = Path("reports/stress_results.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        latency = results['latency']
        throughput = results['throughput']
        performance = results['performance']
        
        report_content = f"""# Load Test Results - 10x Volume Stress Test

**Test Date:** {results['timestamp'][:19]}  
**Test Run ID:** {results['test_run_id']}  
**Target Volume:** {results['configuration']['target_orders']} orders (10x normal)  
**Performance Grade:** {performance['grade']} {'✅' if performance['target_met'] else '❌'}

## 📊 Performance Summary

### Latency Statistics
| Metric | Value |
|--------|-------|
| **P95 Latency** | **{latency['p95_ms']:.1f} ms** |
| P99 Latency | {latency['p99_ms']:.1f} ms |
| Mean Latency | {latency['mean_ms']:.1f} ms |
| Median Latency | {latency['median_ms']:.1f} ms |
| Min Latency | {latency['min_ms']:.1f} ms |
| Max Latency | {latency['max_ms']:.1f} ms |

### Throughput Statistics
| Metric | Value |
|--------|-------|
| Total Orders | {throughput['total_orders']:,} |
| Successful Orders | {throughput['successful_orders']:,} |
| Failed Orders | {throughput['failed_orders']:,} |
| Success Rate | {throughput['success_rate_pct']:.1f}% |
| Orders/Second | {throughput['orders_per_second']:.1f} |
| Total Duration | {throughput['total_duration_sec']:.1f} seconds |

## 🎯 Performance Assessment

**Target:** P95 latency < 500ms  
**Actual:** {latency['p95_ms']:.1f} ms  
**Result:** {'✅ PASS' if performance['target_met'] else '❌ FAIL'}

### Key Findings
- **Order Processing:** {'Excellent' if latency['p95_ms'] < 200 else 'Good' if latency['p95_ms'] < 500 else 'Needs Improvement'}
- **System Stability:** {'Stable' if throughput['success_rate_pct'] > 95 else 'Unstable'}
- **Throughput:** {throughput['orders_per_second']:.0f} orders/second
- **Error Rate:** {(throughput['failed_orders'] / throughput['total_orders'] * 100):.2f}%

## 📈 Latency Distribution

```
P50: {latency['median_ms']:.1f} ms  ████████████████████████████████████████████████████ 50%
P95: {latency['p95_ms']:.1f} ms  ████████████████████████████████████████████████████████████████████████████████████████████████ 95%
P99: {latency['p99_ms']:.1f} ms  ████████████████████████████████████████████████████████████████████████████████████████████████████████████ 99%
```

## 🔍 Analysis

### Performance Characteristics
- **Low Latency Orders:** {len([l for l in self.latency_measurements if l < 100])}/{len(self.latency_measurements)} ({len([l for l in self.latency_measurements if l < 100])/len(self.latency_measurements)*100:.1f}%) under 100ms
- **High Latency Orders:** {len([l for l in self.latency_measurements if l > 1000])}/{len(self.latency_measurements)} ({len([l for l in self.latency_measurements if l > 1000])/len(self.latency_measurements)*100:.1f}%) over 1000ms

### Recommendations
{"- ✅ System ready for 10x volume load" if performance['target_met'] else "- ❌ Performance optimization needed before production"}
{"- ✅ Latency within acceptable range" if latency['p95_ms'] < 500 else "- ⚠️ Consider optimizing order routing pipeline"}
{"- ✅ High success rate indicates stable processing" if throughput['success_rate_pct'] > 95 else "- ⚠️ Error rate too high - investigate failure causes"}

## 🛠️ Test Configuration

- **Database:** {self.db_path}
- **Concurrent Workers:** Multiple (ThreadPoolExecutor)
- **Order Types:** Market, Limit, Stop
- **Symbols:** {', '.join(self.test_symbols)}
- **Strategies:** {', '.join(self.strategies)}

---

*Generated by Mech-Exo Load Testing Framework*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Report generated: {report_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Load Test - 10x Volume Stress Test")
    parser.add_argument("--orders", type=int, default=5000, 
                       help="Number of orders to generate (default: 5000)")
    parser.add_argument("--workers", type=int, default=50,
                       help="Maximum concurrent workers (default: 50)")
    parser.add_argument("--db-path", type=str, default="data/load_test.duckdb",
                       help="Database path for testing")
    
    args = parser.parse_args()
    
    # Initialize and run load test
    tester = OrderLoadTester(args.db_path)
    
    try:
        results = tester.run_load_test(args.orders, args.workers)
        
        # Print summary
        logger.info("\n🎉 LOAD TEST COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"P95 Latency: {results['latency']['p95_ms']:.1f} ms")
        logger.info(f"Success Rate: {results['throughput']['success_rate_pct']:.1f}%")
        logger.info(f"Throughput: {results['throughput']['orders_per_second']:.1f} orders/sec")
        logger.info(f"Performance: {results['performance']['grade']}")
        
        if results['performance']['target_met']:
            logger.info("✅ STRESS TEST PASSED - System ready for 10x load!")
        else:
            logger.warning("❌ STRESS TEST FAILED - Performance optimization needed")
        
        logger.info(f"📄 Detailed report: reports/stress_results.md")
        
    except Exception as e:
        logger.error(f"❌ Load test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())