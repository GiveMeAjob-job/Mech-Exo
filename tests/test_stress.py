"""
Stress Test Suite

Tests system performance under high load conditions.
Validates that p95 latency remains under 500ms with 10x volume.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_test_orders import OrderLoadTester


class TestStressPerformance:
    """Test system performance under stress"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_moderate_load_performance(self, temp_db):
        """Test performance with moderate load (1000 orders)"""
        tester = OrderLoadTester(temp_db)
        
        # Run with moderate load
        results = tester.run_load_test(num_orders=1000, max_workers=20)
        
        # Validate results
        assert results['performance']['target_met'], f"P95 latency {results['latency']['p95_ms']:.1f}ms exceeds 500ms threshold"
        assert results['throughput']['success_rate_pct'] > 95, f"Success rate {results['throughput']['success_rate_pct']:.1f}% too low"
        assert results['throughput']['orders_per_second'] > 50, f"Throughput {results['throughput']['orders_per_second']:.1f} orders/sec too low"
    
    def test_high_load_performance(self, temp_db):
        """Test performance with high load (2500 orders)"""
        tester = OrderLoadTester(temp_db)
        
        # Run with high load
        results = tester.run_load_test(num_orders=2500, max_workers=30)
        
        # Validate results
        assert results['performance']['target_met'], f"P95 latency {results['latency']['p95_ms']:.1f}ms exceeds 500ms threshold"
        assert results['throughput']['success_rate_pct'] > 90, f"Success rate {results['throughput']['success_rate_pct']:.1f}% too low for high load"
        assert results['latency']['mean_ms'] < 200, f"Mean latency {results['latency']['mean_ms']:.1f}ms too high"
    
    @pytest.mark.slow
    def test_extreme_load_performance(self, temp_db):
        """Test performance with extreme load (5000 orders = 10x normal)"""
        tester = OrderLoadTester(temp_db)
        
        # Run with extreme load (10x normal volume)
        results = tester.run_load_test(num_orders=5000, max_workers=50)
        
        # Validate critical performance metrics
        assert results['performance']['target_met'], f"CRITICAL: P95 latency {results['latency']['p95_ms']:.1f}ms exceeds 500ms threshold"
        assert results['throughput']['success_rate_pct'] > 85, f"Success rate {results['throughput']['success_rate_pct']:.1f}% too low for extreme load"
        
        # Log detailed results for analysis
        print(f"\n🔍 EXTREME LOAD TEST RESULTS:")
        print(f"   P95 Latency: {results['latency']['p95_ms']:.1f} ms")
        print(f"   P99 Latency: {results['latency']['p99_ms']:.1f} ms")
        print(f"   Success Rate: {results['throughput']['success_rate_pct']:.1f}%")
        print(f"   Throughput: {results['throughput']['orders_per_second']:.1f} orders/sec")
        print(f"   Total Duration: {results['throughput']['total_duration_sec']:.1f} seconds")
    
    def test_database_integrity_under_load(self, temp_db):
        """Test that database remains consistent under concurrent load"""
        tester = OrderLoadTester(temp_db)
        
        # Run load test
        results = tester.run_load_test(num_orders=1000, max_workers=25)
        
        # Check database integrity
        from mech_exo.datasource.storage import DataStorage
        storage = DataStorage(temp_db)
        
        # Verify order count
        order_count = storage.conn.execute("SELECT COUNT(*) FROM load_test_orders").fetchone()[0]
        assert order_count == results['throughput']['total_orders'], "Order count mismatch"
        
        # Verify fill count
        fill_count = storage.conn.execute("SELECT COUNT(*) FROM load_test_fills").fetchone()[0]
        successful_orders = results['throughput']['successful_orders']
        
        # Fill count should be close to successful orders (some might be rejected)
        assert fill_count <= successful_orders, "More fills than successful orders"
        assert fill_count >= successful_orders * 0.9, "Too few fills generated"
        
        # Verify no orphaned fills
        orphaned_fills = storage.conn.execute("""
            SELECT COUNT(*) FROM load_test_fills f
            LEFT JOIN load_test_orders o ON f.order_id = o.order_id
            WHERE o.order_id IS NULL
        """).fetchone()[0]
        assert orphaned_fills == 0, f"Found {orphaned_fills} orphaned fills"
        
        storage.close()
    
    def test_latency_distribution(self, temp_db):
        """Test that latency distribution is reasonable"""
        tester = OrderLoadTester(temp_db)
        
        # Run load test
        results = tester.run_load_test(num_orders=1000, max_workers=20)
        
        latency = results['latency']
        
        # Check latency distribution characteristics
        assert latency['min_ms'] < latency['median_ms'] < latency['p95_ms'] < latency['max_ms'], "Latency distribution order incorrect"
        assert latency['mean_ms'] < latency['p95_ms'], "Mean should be less than P95"
        assert latency['p95_ms'] < latency['p99_ms'], "P95 should be less than P99"
        
        # Check for reasonable spread
        latency_range = latency['max_ms'] - latency['min_ms']
        assert latency_range < 5000, f"Latency range {latency_range:.1f}ms too large - indicates performance issues"
    
    def test_concurrent_worker_scaling(self, temp_db):
        """Test performance with different worker counts"""
        tester = OrderLoadTester(temp_db)
        
        test_configs = [
            (500, 10),   # Low concurrency
            (500, 25),   # Medium concurrency  
            (500, 50),   # High concurrency
        ]
        
        results = []
        
        for orders, workers in test_configs:
            # Reset measurements for each test
            tester.latency_measurements = []
            tester.error_count = 0
            tester.total_orders = 0
            
            result = tester.run_load_test(num_orders=orders, max_workers=workers)
            results.append((workers, result))
        
        # Verify all configurations meet performance requirements
        for workers, result in results:
            assert result['performance']['target_met'], f"Worker config {workers} failed performance test"
            assert result['throughput']['success_rate_pct'] > 90, f"Worker config {workers} has low success rate"


if __name__ == "__main__":
    # Run stress tests directly
    pytest.main([__file__, "-v", "--tb=short"])