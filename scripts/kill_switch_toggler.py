#!/usr/bin/env python3
"""
Kill Switch Toggler for Chaos Testing
Toggles kill switch every 3 minutes to test recovery time
"""

import subprocess
import time
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kill_switch_toggler')


class KillSwitchToggler:
    """Manages kill switch toggle operations with timing and recovery metrics"""
    
    def __init__(self, interval_minutes: int = 3):
        self.interval_seconds = interval_minutes * 60
        self.toggle_count = 0
        self.recovery_times = []
        self.failed_toggles = []
        self.start_time = datetime.utcnow()
        
        # Ensure logs directory exists
        Path('/tmp/chaos_logs').mkdir(exist_ok=True)
    
    def execute_toggle(self) -> Dict:
        """Execute a single kill switch toggle cycle"""
        self.toggle_count += 1
        toggle_start = time.time()
        
        logger.info(f"Starting kill switch toggle #{self.toggle_count}")
        
        result = {
            'toggle_id': self.toggle_count,
            'timestamp': datetime.utcnow().isoformat(),
            'disable_time': None,
            'enable_time': None,
            'total_recovery_time': None,
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Disable kill switch
            disable_start = time.time()
            disable_result = subprocess.run([
                'python', 'mech_exo/cli.py', 'killswitch', 'disable',
                '--reason', f'chaos_test_{self.toggle_count}'
            ], capture_output=True, text=True, timeout=60)
            
            disable_duration = time.time() - disable_start
            result['disable_time'] = disable_duration
            
            if disable_result.returncode != 0:
                result['error'] = f"Disable failed: {disable_result.stderr}"
                logger.error(f"Toggle #{self.toggle_count} disable failed: {disable_result.stderr}")
                self.failed_toggles.append(result)
                return result
            
            logger.info(f"Toggle #{self.toggle_count} disabled in {disable_duration:.2f}s")
            
            # Wait 30 seconds (simulating downtime)
            logger.info(f"Toggle #{self.toggle_count} waiting 30s before re-enable...")
            time.sleep(30)
            
            # Step 2: Re-enable kill switch
            enable_start = time.time()
            enable_result = subprocess.run([
                'python', 'mech_exo/cli.py', 'killswitch', 'enable',
                '--reason', 'chaos_test_recovery'
            ], capture_output=True, text=True, timeout=60)
            
            enable_duration = time.time() - enable_start
            result['enable_time'] = enable_duration
            
            if enable_result.returncode != 0:
                result['error'] = f"Enable failed: {enable_result.stderr}"
                logger.error(f"Toggle #{self.toggle_count} enable failed: {enable_result.stderr}")
                self.failed_toggles.append(result)
                return result
            
            # Step 3: Verify system is operational
            total_recovery = time.time() - toggle_start
            result['total_recovery_time'] = total_recovery
            result['success'] = True
            
            self.recovery_times.append(total_recovery)
            
            logger.info(f"Toggle #{self.toggle_count} completed successfully in {total_recovery:.2f}s")
            
            # Check if recovery time exceeds threshold
            if total_recovery > 60:  # 60 second SLO
                logger.warning(f"Toggle #{self.toggle_count} recovery time {total_recovery:.2f}s exceeds 60s SLO")
            
            # Verify system health after recovery
            self._verify_system_health()
            
        except subprocess.TimeoutExpired:
            result['error'] = "Command timeout"
            logger.error(f"Toggle #{self.toggle_count} timed out")
            self.failed_toggles.append(result)
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Toggle #{self.toggle_count} failed with exception: {e}")
            self.failed_toggles.append(result)
        
        return result
    
    def _verify_system_health(self):
        """Verify system health after kill switch recovery"""
        try:
            # Check if trading is enabled via health endpoint
            health_result = subprocess.run([
                'curl', '-s', 'http://localhost:8000/api/health'
            ], capture_output=True, text=True, timeout=10)
            
            if health_result.returncode == 0:
                try:
                    health_data = json.loads(health_result.stdout)
                    if health_data.get('trading_enabled', False):
                        logger.info(f"System health verified - trading enabled")
                    else:
                        logger.warning(f"System health check - trading still disabled")
                except json.JSONDecodeError:
                    logger.warning("System health check - invalid JSON response")
            else:
                logger.warning("System health check failed - endpoint not accessible")
                
        except Exception as e:
            logger.warning(f"System health verification failed: {e}")
    
    def get_statistics(self) -> Dict:
        """Get current toggle statistics"""
        if self.recovery_times:
            avg_recovery = sum(self.recovery_times) / len(self.recovery_times)
            max_recovery = max(self.recovery_times)
            min_recovery = min(self.recovery_times)
        else:
            avg_recovery = max_recovery = min_recovery = 0
        
        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'total_toggles': self.toggle_count,
            'successful_toggles': len(self.recovery_times),
            'failed_toggles': len(self.failed_toggles),
            'average_recovery_time': avg_recovery,
            'max_recovery_time': max_recovery,
            'min_recovery_time': min_recovery,
            'success_rate': len(self.recovery_times) / max(1, self.toggle_count) * 100,
            'slo_violations': len([t for t in self.recovery_times if t > 60]),
            'runtime_seconds': runtime
        }
    
    def log_statistics(self):
        """Log current statistics"""
        stats = self.get_statistics()
        
        logger.info("=" * 50)
        logger.info("KILL SWITCH TOGGLE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total toggles: {stats['total_toggles']}")
        logger.info(f"Successful: {stats['successful_toggles']}")
        logger.info(f"Failed: {stats['failed_toggles']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"Average recovery: {stats['average_recovery_time']:.2f}s")
        logger.info(f"Max recovery: {stats['max_recovery_time']:.2f}s")
        logger.info(f"Min recovery: {stats['min_recovery_time']:.2f}s")
        logger.info(f"SLO violations (>60s): {stats['slo_violations']}")
        logger.info(f"Runtime: {stats['runtime_seconds']:.0f}s")
        logger.info("=" * 50)
    
    def save_results(self):
        """Save detailed results to file"""
        results = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'interval_seconds': self.interval_seconds
            },
            'statistics': self.get_statistics(),
            'recovery_times': self.recovery_times,
            'failed_toggles': self.failed_toggles
        }
        
        results_file = Path('/tmp/chaos_logs/kill_switch_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Kill switch results saved to {results_file}")
    
    def run_continuous(self, duration_hours: int = 24):
        """Run continuous kill switch toggling for specified duration"""
        end_time = time.time() + (duration_hours * 3600)
        next_toggle = time.time()
        
        logger.info(f"Starting continuous kill switch toggling for {duration_hours} hours")
        logger.info(f"Toggle interval: {self.interval_seconds} seconds ({self.interval_seconds/60} minutes)")
        
        try:
            while time.time() < end_time:
                if time.time() >= next_toggle:
                    # Execute toggle
                    result = self.execute_toggle()
                    
                    # Schedule next toggle
                    next_toggle = time.time() + self.interval_seconds
                    
                    # Log statistics every 10 toggles
                    if self.toggle_count % 10 == 0:
                        self.log_statistics()
                    
                    # Save results periodically
                    if self.toggle_count % 20 == 0:
                        self.save_results()
                
                # Sleep for 30 seconds before checking again
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("Kill switch toggling interrupted by user")
        except Exception as e:
            logger.error(f"Kill switch toggling failed: {e}")
        finally:
            self.log_statistics()
            self.save_results()
    
    def run_count(self, count: int):
        """Run a specific number of kill switch toggles"""
        logger.info(f"Starting {count} kill switch toggles")
        
        try:
            for i in range(count):
                result = self.execute_toggle()
                
                # Wait interval between toggles (except for last one)
                if i < count - 1:
                    logger.info(f"Waiting {self.interval_seconds}s before next toggle...")
                    time.sleep(self.interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Kill switch toggling interrupted by user")
        except Exception as e:
            logger.error(f"Kill switch toggling failed: {e}")
        finally:
            self.log_statistics()
            self.save_results()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kill Switch Toggler for Chaos Testing')
    parser.add_argument('--interval', type=int, default=3, help='Toggle interval in minutes (default: 3)')
    parser.add_argument('--duration', type=int, help='Duration in hours for continuous mode')
    parser.add_argument('--count', type=int, help='Number of toggles to execute')
    parser.add_argument('--test', action='store_true', help='Run single test toggle')
    
    args = parser.parse_args()
    
    if args.test:
        # Single test toggle
        toggler = KillSwitchToggler(interval_minutes=1)
        result = toggler.execute_toggle()
        toggler.log_statistics()
        print(f"Test result: {result}")
    elif args.count:
        # Run specific number of toggles
        toggler = KillSwitchToggler(interval_minutes=args.interval)
        toggler.run_count(args.count)
    elif args.duration:
        # Run continuous for specified duration
        toggler = KillSwitchToggler(interval_minutes=args.interval)
        toggler.run_continuous(duration_hours=args.duration)
    else:
        # Default: run for 24 hours
        toggler = KillSwitchToggler(interval_minutes=args.interval)
        toggler.run_continuous(duration_hours=24)


if __name__ == "__main__":
    main()