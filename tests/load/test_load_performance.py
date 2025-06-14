#!/usr/bin/env python3
"""
Load Performance Tests for Mech-Exo Risk Control
Validates system can handle ≥50 req/sec for sustained periods
"""

import pytest
import asyncio
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import sys
import os

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [LOAD TEST] %(message)s'
)
logger = logging.getLogger(__name__)


class LoadTestRunner:
    """Load test runner for pytest integration"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or str(Path(__file__).parent / "target_rate.yaml")
        self.config = self._load_config()
        self.results = {}
        self.start_time = None
        self.load_process = None
        self.chaos_process = None
        self.metrics_process = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {str(e)}")
            # Return default config
            return {
                'target_rate': 50,
                'ramp_duration': 60,
                'test_duration': 300,  # 5 minutes for testing
                'base_url': 'http://localhost:8050',
                'paper_trading': True,
                'error_threshold': 0.03,
                'pass_criteria': {
                    'max_error_rate': 0.01,
                    'max_95p_latency_ms': 400,
                    'min_success_rate': 0.99
                }
            }
    
    def start_load_runner(self, rate: int, duration: int, ramp: int = 60) -> subprocess.Popen:
        """Start the load runner process"""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "chaos" / "load_runner.py"
        
        cmd = [
            sys.executable, str(script_path),
            '--rate', str(rate),
            '--duration', str(duration),
            '--ramp', str(ramp),
            '--url', self.config.get('base_url', 'http://localhost:8050'),
            '--paper'
        ]
        
        logger.info(f"Starting load runner: {' '.join(cmd)}")
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def start_chaos_monkey(self, duration: int) -> subprocess.Popen:
        """Start the chaos monkey process"""
        script_path = Path(__file__).parent.parent / "chaos" / "kill_switch_toggler.py"
        
        cmd = [
            sys.executable, str(script_path),
            '--duration', str(duration),
            '--url', self.config.get('base_url', 'http://localhost:8050'),
            '--min-interval', '3',
            '--max-interval', '5',
            '--max-downtime', '60'
        ]
        
        logger.info(f"Starting chaos monkey: {' '.join(cmd)}")
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def start_metrics_exporter(self, duration: int) -> subprocess.Popen:
        """Start the metrics exporter process"""
        script_path = Path(__file__).parent.parent.parent / "prometheus" / "load_metrics.py"
        
        cmd = [
            sys.executable, str(script_path),
            '--duration', str(duration),
            '--poll-interval', '15',
            '--output', 'load_test_metrics.prom'
        ]
        
        logger.info(f"Starting metrics exporter: {' '.join(cmd)}")
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def wait_for_processes(self, timeout: int = None):
        """Wait for all processes to complete"""
        processes = []
        if self.load_process:
            processes.append(('load_runner', self.load_process))
        if self.chaos_process:
            processes.append(('chaos_monkey', self.chaos_process))
        if self.metrics_process:
            processes.append(('metrics_exporter', self.metrics_process))
        
        logger.info(f"Waiting for {len(processes)} processes to complete")
        
        for name, process in processes:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                if process.returncode == 0:
                    logger.info(f"✅ {name} completed successfully")
                else:
                    logger.error(f"❌ {name} failed with code {process.returncode}")
                    if stderr:
                        logger.error(f"{name} stderr: {stderr}")
            except subprocess.TimeoutExpired:
                logger.warning(f"⏰ {name} timed out, terminating")
                process.terminate()
                process.wait()
    
    def cleanup_processes(self):
        """Clean up running processes"""
        processes = [
            ('load_runner', self.load_process),
            ('chaos_monkey', self.chaos_process),
            ('metrics_exporter', self.metrics_process)
        ]
        
        for name, process in processes:
            if process and process.poll() is None:
                logger.info(f"Terminating {name}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}")
                    process.kill()
    
    def collect_results(self) -> Dict[str, Any]:
        """Collect and analyze test results"""
        results = {
            'test_config': self.config,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now().isoformat(),
            'load_results': None,
            'chaos_results': None,
            'metrics_results': None,
            'pass_criteria_met': False
        }
        
        # Load test results
        load_results_files = list(Path(".").glob("load_test_results_*.json"))
        if load_results_files:
            latest_load_file = max(load_results_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_load_file, 'r') as f:
                    results['load_results'] = json.load(f)
                logger.info(f"Loaded results from {latest_load_file}")
            except Exception as e:
                logger.error(f"Failed to load results from {latest_load_file}: {str(e)}")
        
        # Chaos test results
        chaos_results_files = list(Path(".").glob("chaos_test_report_*.json"))
        if chaos_results_files:
            latest_chaos_file = max(chaos_results_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_chaos_file, 'r') as f:
                    results['chaos_results'] = json.load(f)
                logger.info(f"Loaded chaos results from {latest_chaos_file}")
            except Exception as e:
                logger.error(f"Failed to load chaos results from {latest_chaos_file}: {str(e)}")
        
        # Check pass criteria
        results['pass_criteria_met'] = self._check_pass_criteria(results)
        
        return results
    
    def _check_pass_criteria(self, results: Dict[str, Any]) -> bool:
        """Check if test results meet pass criteria"""
        criteria = self.config.get('pass_criteria', {})
        
        if not results.get('load_results'):
            logger.error("No load test results available")
            return False
        
        load_metrics = results['load_results'].get('metrics', {})
        
        # Check error rate
        error_rate = load_metrics.get('error_rate', 1.0)
        max_error_rate = criteria.get('max_error_rate', 0.01)
        if error_rate > max_error_rate:
            logger.error(f"Error rate {error_rate:.2%} exceeds limit {max_error_rate:.2%}")
            return False
        
        # Check latency
        avg_latency_ms = load_metrics.get('avg_latency', 0) * 1000
        max_latency_ms = criteria.get('max_95p_latency_ms', 400)
        if avg_latency_ms > max_latency_ms:
            logger.error(f"Average latency {avg_latency_ms:.1f}ms exceeds limit {max_latency_ms}ms")
            return False
        
        # Check success rate
        total_requests = load_metrics.get('total_requests', 0)
        successful_requests = load_metrics.get('successful_requests', 0)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        min_success_rate = criteria.get('min_success_rate', 0.99)
        if success_rate < min_success_rate:
            logger.error(f"Success rate {success_rate:.2%} below minimum {min_success_rate:.2%}")
            return False
        
        logger.info("✅ All pass criteria met")
        return True
    
    def run_load_test(self, rate: int, duration: int, ramp: int = 60, enable_chaos: bool = False) -> Dict[str, Any]:
        """Run complete load test"""
        logger.info(f"🏋️ Starting load test: {rate} req/s for {duration}s")
        
        self.start_time = datetime.now()
        
        try:
            # Start processes
            self.load_process = self.start_load_runner(rate, duration, ramp)
            
            if enable_chaos:
                self.chaos_process = self.start_chaos_monkey(duration)
            
            self.metrics_process = self.start_metrics_exporter(duration + 120)  # Extra time for cleanup
            
            # Wait for processes
            self.wait_for_processes(duration + 300)  # 5 minute buffer
            
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed with exception: {str(e)}")
        finally:
            # Cleanup
            self.cleanup_processes()
        
        # Collect and return results
        results = self.collect_results()
        
        # Save consolidated results
        results_file = f"consolidated_load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_file}")
        
        return results


# Pytest fixtures and test functions

@pytest.fixture
def load_test_runner():
    """Pytest fixture for load test runner"""
    return LoadTestRunner()


@pytest.mark.load
def test_light_load_performance(load_test_runner):
    """Test system performance under light load (10 req/s)"""
    results = load_test_runner.run_load_test(
        rate=10,
        duration=300,  # 5 minutes
        ramp=30
    )
    
    assert results['load_results'] is not None, "Load test results not available"
    assert results['pass_criteria_met'], "Light load test failed to meet pass criteria"


@pytest.mark.load
def test_target_load_performance(load_test_runner):
    """Test system performance under target load (50 req/s)"""
    results = load_test_runner.run_load_test(
        rate=50,
        duration=1800,  # 30 minutes
        ramp=60
    )
    
    assert results['load_results'] is not None, "Load test results not available"
    assert results['pass_criteria_met'], "Target load test failed to meet pass criteria"


@pytest.mark.load
@pytest.mark.chaos
def test_target_load_with_chaos(load_test_runner):
    """Test system performance under target load with chaos monkey"""
    results = load_test_runner.run_load_test(
        rate=50,
        duration=1800,  # 30 minutes
        ramp=60,
        enable_chaos=True
    )
    
    assert results['load_results'] is not None, "Load test results not available"
    assert results['chaos_results'] is not None, "Chaos test results not available"
    assert results['pass_criteria_met'], "Load test with chaos failed to meet pass criteria"
    
    # Check chaos monkey specific criteria
    chaos_results = results['chaos_results']
    total_toggles = chaos_results.get('total_toggles', 0)
    successful_recoveries = chaos_results.get('successful_recoveries', 0)
    
    if total_toggles > 0:
        recovery_rate = successful_recoveries / total_toggles
        assert recovery_rate >= 0.95, f"Recovery rate {recovery_rate:.1%} below 95% threshold"


@pytest.mark.load
@pytest.mark.endurance
def test_endurance_load(load_test_runner):
    """Test system endurance under sustained load (1 hour)"""
    results = load_test_runner.run_load_test(
        rate=50,
        duration=3600,  # 1 hour
        ramp=60,
        enable_chaos=True
    )
    
    assert results['load_results'] is not None, "Load test results not available"
    assert results['pass_criteria_met'], "Endurance test failed to meet pass criteria"


@pytest.mark.load
@pytest.mark.spike
def test_spike_load_performance(load_test_runner):
    """Test system performance under spike load (100 req/s)"""
    results = load_test_runner.run_load_test(
        rate=100,
        duration=300,  # 5 minutes
        ramp=10
    )
    
    assert results['load_results'] is not None, "Load test results not available"
    
    # Spike test has more relaxed criteria
    load_metrics = results['load_results'].get('metrics', {})
    error_rate = load_metrics.get('error_rate', 1.0)
    
    # Allow higher error rate for spike tests
    assert error_rate <= 0.05, f"Spike test error rate {error_rate:.2%} exceeds 5% threshold"


def main():
    """Main function for running tests directly"""
    import sys
    
    parser = argparse.ArgumentParser(description='Load performance tests')
    parser.add_argument('--rate', type=int, default=50, help='Request rate (req/sec)')
    parser.add_argument('--duration', type=int, default=300, help='Test duration (seconds)')
    parser.add_argument('--ramp', type=int, default=60, help='Ramp duration (seconds)')
    parser.add_argument('--chaos', action='store_true', help='Enable chaos monkey')
    
    args = parser.parse_args()
    
    # Run test directly
    runner = LoadTestRunner()
    results = runner.run_load_test(
        rate=args.rate,
        duration=args.duration,
        ramp=args.ramp,
        enable_chaos=args.chaos
    )
    
    if results['pass_criteria_met']:
        logger.info("🎉 Load test PASSED")
        return 0
    else:
        logger.error("❌ Load test FAILED")
        return 1


if __name__ == "__main__":
    import argparse
    sys.exit(main())