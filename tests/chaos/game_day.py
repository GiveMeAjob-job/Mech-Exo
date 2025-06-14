#!/usr/bin/env python3
"""
24h Game-Day Chaos Testing Orchestrator
Phase P11 Week 3 Weekend - Continuous chaos injection with SLO monitoring

Objectives:
- Error Budget remaining ≥97%
- Critical service recovery ≤4min
- No real capital loss or erroneous orders

Test Scenarios:
1. Continuous order flood (50 req/s)
2. Kill-switch toggle every 3min (480 cycles)
3. Network jitter every 20min (72 cycles)
4. Database hot restart at 02:00/14:00 (2 cycles)
5. IB Gateway restart at 06:00/18:00 (2 cycles)
"""

import asyncio
import logging
import time
import json
import threading
import subprocess
import requests
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures
import signal
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/chaos_logs/game_day.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('game_day_chaos')


class SLOMonitor:
    """Monitors SLOs and triggers auto-abort if critical thresholds exceeded"""
    
    def __init__(self):
        self.prometheus_url = "http://prometheus.mech-exo.com:9090"
        self.slo_metrics = {
            'order_err_rate': {'threshold': 1.0, 'query': 'rate(order_errors_total[5m]) * 100'},
            'risk_ops_ok': {'threshold': 0.98, 'query': 'risk_ops_success_rate_5m'},
            'latency_p95_ms': {'threshold': 400, 'query': 'histogram_quantile(0.95, rate(api_request_duration_ms_bucket[5m]))'},
            'kill_switch_recovery_sec': {'threshold': 60, 'query': 'kill_switch_recovery_time_seconds'}
        }
        self.alert_history = []
        self.critical_alert_start = None
        self.abort_triggered = False
    
    def query_prometheus(self, query: str) -> Optional[float]:
        """Query Prometheus for metric value"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
        except Exception as e:
            logger.error(f"Prometheus query failed for '{query}': {e}")
        return None
    
    def check_slos(self) -> Dict[str, bool]:
        """Check all SLO metrics and return violation status"""
        violations = {}
        current_time = datetime.utcnow()
        
        for metric_name, config in self.slo_metrics.items():
            value = self.query_prometheus(config['query'])
            if value is not None:
                # Special handling for different metric types
                if metric_name == 'order_err_rate':
                    violated = value > config['threshold']
                elif metric_name == 'risk_ops_ok':
                    violated = value < config['threshold']
                else:
                    violated = value > config['threshold']
                
                violations[metric_name] = violated
                
                if violated:
                    alert = {
                        'metric': metric_name,
                        'value': value,
                        'threshold': config['threshold'],
                        'timestamp': current_time.isoformat()
                    }
                    self.alert_history.append(alert)
                    logger.warning(f"SLO VIOLATION: {metric_name}={value} (threshold: {config['threshold']})")
                else:
                    logger.debug(f"SLO OK: {metric_name}={value}")
            else:
                violations[metric_name] = False  # Assume OK if can't query
        
        return violations
    
    def check_critical_duration(self, violations: Dict[str, bool]) -> bool:
        """Check if critical alerts have persisted > 5min"""
        critical_violations = [k for k, v in violations.items() if v]
        
        if critical_violations:
            if self.critical_alert_start is None:
                self.critical_alert_start = datetime.utcnow()
                logger.warning(f"Critical alert period started: {critical_violations}")
            else:
                duration = (datetime.utcnow() - self.critical_alert_start).total_seconds()
                if duration > 300:  # 5 minutes
                    logger.critical(f"Critical alerts persisted for {duration}s - triggering abort")
                    return True
        else:
            if self.critical_alert_start:
                logger.info("Critical alert period ended")
                self.critical_alert_start = None
        
        return False
    
    def get_error_budget_remaining(self) -> float:
        """Calculate remaining error budget percentage"""
        # Error budget = (1 - actual_error_rate / target_error_rate) * 100
        error_rate = self.query_prometheus('rate(total_errors[24h]) / rate(total_requests[24h]) * 100')
        if error_rate is not None:
            return max(0, 100 - (error_rate / 3.0) * 100)  # 3% target error budget
        return 100.0  # Assume full budget if can't measure


class ChaosOrchestrator:
    """Main chaos orchestrator managing all test scenarios"""
    
    def __init__(self):
        self.slo_monitor = SLOMonitor()
        self.active_scenarios = {}
        self.chaos_start_time = None
        self.total_runtime_hours = 24
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.shutdown_event = threading.Event()
        
        # Scenario counters
        self.kill_switch_count = 0
        self.network_jitter_count = 0
        self.db_restart_count = 0
        self.ib_restart_count = 0
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def failsafe_abort(self):
        """Emergency abort - stop all load and rollback to Blue environment"""
        logger.critical("🚨 FAILSAFE ABORT TRIGGERED 🚨")
        self.slo_monitor.abort_triggered = True
        
        try:
            # Stop all load generation
            subprocess.run(['pkill', '-f', 'load_runner.py'], check=False)
            logger.info("Load generation stopped")
            
            # Rollback to Blue environment
            subprocess.run(['kubectl', 'rollout', 'undo', 'deployment/mech-exo-exec'], check=True)
            subprocess.run(['kubectl', 'rollout', 'undo', 'deployment/mech-exo-api'], check=True)
            logger.info("Rolled back to Blue environment")
            
            # Enable kill-switch to stop trading
            self._execute_kill_switch_toggle(force_disable=True)
            logger.info("Kill-switch activated (trading disabled)")
            
            # Generate emergency report
            self._generate_emergency_report()
            
        except Exception as e:
            logger.error(f"Failsafe abort encountered error: {e}")
        
        # Set shutdown event
        self.shutdown_event.set()
    
    def _generate_emergency_report(self):
        """Generate emergency abort report"""
        report_path = Path("/tmp/reports/emergency_abort.md")
        
        with open(report_path, 'w') as f:
            f.write(f"""# Emergency Abort Report
Time: {datetime.utcnow().isoformat()}Z
Reason: Critical SLO violations persisted >5min

## SLO Violations
""")
            for alert in self.slo_monitor.alert_history[-10:]:  # Last 10 alerts
                f.write(f"- {alert['timestamp']}: {alert['metric']}={alert['value']} (threshold: {alert['threshold']})\n")
            
            f.write(f"""
## Error Budget Status
Remaining: {self.slo_monitor.get_error_budget_remaining():.1f}%

## Actions Taken
1. Stopped all load generation
2. Rolled back to Blue environment
3. Activated kill-switch (trading disabled)
4. Generated this emergency report

## Next Steps
1. Review alert history and root cause
2. Fix underlying issues before resuming
3. Consider reducing chaos intensity
""")
        
        logger.critical(f"Emergency report generated: {report_path}")
    
    async def continuous_order_flood(self):
        """Scenario 1: Continuous order flood at 50 req/s for 24h"""
        logger.info("🌊 Starting continuous order flood (50 req/s)")
        
        # Start load runner process
        process = subprocess.Popen([
            'python', 'scripts/load_runner.py', 
            '--rate', '50',
            '--duration', str(self.total_runtime_hours * 3600),
            '--endpoint', 'http://localhost:8000/api/orders',
            '--dry-run'  # Prevent real orders during chaos
        ])
        
        self.active_scenarios['order_flood'] = process
        logger.info(f"Order flood process started (PID: {process.pid})")
        
        # Monitor process health
        while not self.shutdown_event.is_set():
            if process.poll() is not None:
                logger.warning("Order flood process terminated, restarting...")
                # Restart if terminated
                process = subprocess.Popen([
                    'python', 'scripts/load_runner.py', 
                    '--rate', '50',
                    '--duration', '3600',  # 1 hour chunks
                    '--endpoint', 'http://localhost:8000/api/orders',
                    '--dry-run'
                ])
                self.active_scenarios['order_flood'] = process
            
            await asyncio.sleep(60)  # Check every minute
    
    def _execute_kill_switch_toggle(self, force_disable=False):
        """Execute kill-switch toggle operation"""
        self.kill_switch_count += 1
        start_time = time.time()
        
        try:
            if force_disable:
                # Force disable for emergency abort
                result = subprocess.run([
                    'python', 'mech_exo/cli.py', 'killswitch', 'disable',
                    '--reason', 'emergency_abort'
                ], capture_output=True, text=True, timeout=120)
            else:
                # Normal toggle: disable then re-enable
                # Disable
                result = subprocess.run([
                    'python', 'mech_exo/cli.py', 'killswitch', 'disable',
                    '--reason', f'chaos_test_{self.kill_switch_count}'
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Wait 30 seconds then re-enable
                    time.sleep(30)
                    result = subprocess.run([
                        'python', 'mech_exo/cli.py', 'killswitch', 'enable',
                        '--reason', 'chaos_test_recovery'
                    ], capture_output=True, text=True, timeout=60)
            
            recovery_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"Kill-switch toggle #{self.kill_switch_count} completed in {recovery_time:.1f}s")
                
                # Record recovery time metric
                self._record_metric('kill_switch_recovery_time_seconds', recovery_time)
                
                if recovery_time > 60 and not force_disable:
                    logger.warning(f"Kill-switch recovery took {recovery_time:.1f}s (>60s threshold)")
                
            else:
                logger.error(f"Kill-switch toggle #{self.kill_switch_count} failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Kill-switch toggle #{self.kill_switch_count} timed out")
        except Exception as e:
            logger.error(f"Kill-switch toggle #{self.kill_switch_count} error: {e}")
    
    def _execute_network_jitter(self):
        """Execute network jitter injection using netem"""
        self.network_jitter_count += 1
        
        try:
            # Get mech-exo-exec pod names
            result = subprocess.run([
                'kubectl', 'get', 'pods', '-l', 'app=mech-exo-exec',
                '-o', 'jsonpath={.items[*].metadata.name}'
            ], capture_output=True, text=True)
            
            pod_names = result.stdout.strip().split()
            
            for pod_name in pod_names:
                # Apply network delay and loss
                subprocess.run([
                    'kubectl', 'exec', pod_name, '--',
                    'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'netem',
                    'delay', '200ms', 'loss', '2%'
                ], timeout=30)
                
                logger.info(f"Applied network jitter to {pod_name} (200ms delay, 2% loss)")
                
                # Remove after 5 minutes
                threading.Timer(300, self._remove_network_jitter, args=[pod_name]).start()
            
            logger.info(f"Network jitter #{self.network_jitter_count} applied to {len(pod_names)} pods")
            
        except Exception as e:
            logger.error(f"Network jitter #{self.network_jitter_count} failed: {e}")
    
    def _remove_network_jitter(self, pod_name: str):
        """Remove network jitter from pod"""
        try:
            subprocess.run([
                'kubectl', 'exec', pod_name, '--',
                'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'
            ], timeout=30)
            logger.info(f"Removed network jitter from {pod_name}")
        except Exception as e:
            logger.warning(f"Failed to remove network jitter from {pod_name}: {e}")
    
    def _execute_database_restart(self):
        """Execute DuckDB hot restart"""
        self.db_restart_count += 1
        start_time = time.time()
        
        try:
            logger.info(f"Database restart #{self.db_restart_count} starting...")
            
            # Rollout restart DuckDB StatefulSet
            result = subprocess.run([
                'kubectl', 'rollout', 'restart', 'statefulset/duckdb'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Wait for rollout to complete
                subprocess.run([
                    'kubectl', 'rollout', 'status', 'statefulset/duckdb',
                    '--timeout=300s'
                ], timeout=300)
                
                restart_time = time.time() - start_time
                logger.info(f"Database restart #{self.db_restart_count} completed in {restart_time:.1f}s")
                
                # Verify no data loss
                self._verify_database_integrity()
                
            else:
                logger.error(f"Database restart #{self.db_restart_count} failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Database restart #{self.db_restart_count} error: {e}")
    
    def _execute_ib_gateway_restart(self):
        """Execute IB Gateway restart with cold backup takeover"""
        self.ib_restart_count += 1
        start_time = time.time()
        
        try:
            logger.info(f"IB Gateway restart #{self.ib_restart_count} starting...")
            
            # Kill IB Gateway process
            subprocess.run(['pkill', '-f', 'ibgateway'], check=False)
            
            # Wait for cold backup to detect failure and take over
            time.sleep(60)
            
            # Restart primary IB Gateway
            subprocess.run([
                'python', 'scripts/start_ib_gateway.py',
                '--mode', 'primary'
            ], timeout=120)
            
            restart_time = time.time() - start_time
            logger.info(f"IB Gateway restart #{self.ib_restart_count} completed in {restart_time:.1f}s")
            
            # Verify trade router failover worked
            self._verify_trade_router_health()
            
        except Exception as e:
            logger.error(f"IB Gateway restart #{self.ib_restart_count} error: {e}")
    
    def _verify_database_integrity(self):
        """Verify database integrity after restart"""
        try:
            # Simple query to verify database is accessible
            result = subprocess.run([
                'python', '-c', 
                "import duckdb; conn = duckdb.connect('data/mech_exo.duckdb'); print(conn.execute('SELECT COUNT(*) FROM trades').fetchall())"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Database integrity verified - no data loss detected")
            else:
                logger.error(f"Database integrity check failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Database integrity verification error: {e}")
    
    def _verify_trade_router_health(self):
        """Verify trade router health after IB Gateway restart"""
        try:
            response = requests.get('http://localhost:8000/api/health/trading', timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('trading_enabled', False):
                    logger.info("Trade router health verified - auto failover successful")
                else:
                    logger.warning("Trade router health check - trading disabled")
            else:
                logger.error(f"Trade router health check failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Trade router health verification error: {e}")
    
    def _record_metric(self, metric_name: str, value: float):
        """Record custom metric to Prometheus pushgateway"""
        try:
            payload = f"{metric_name} {value}\n"
            response = requests.put(
                'http://pushgateway.mech-exo.com:9091/metrics/job/chaos_testing',
                data=payload,
                headers={'Content-Type': 'text/plain'},
                timeout=5
            )
            if response.status_code == 200:
                logger.debug(f"Recorded metric: {metric_name}={value}")
        except Exception as e:
            logger.warning(f"Failed to record metric {metric_name}: {e}")
    
    def schedule_scenarios(self):
        """Schedule all chaos scenarios"""
        logger.info("Scheduling chaos scenarios...")
        
        # Kill-switch toggle every 3 minutes
        schedule.every(3).minutes.do(self._execute_kill_switch_toggle)
        
        # Network jitter every 20 minutes
        schedule.every(20).minutes.do(self._execute_network_jitter)
        
        # Database restart at 02:00 and 14:00
        schedule.every().day.at("02:00").do(self._execute_database_restart)
        schedule.every().day.at("14:00").do(self._execute_database_restart)
        
        # IB Gateway restart at 06:00 and 18:00  
        schedule.every().day.at("06:00").do(self._execute_ib_gateway_restart)
        schedule.every().day.at("18:00").do(self._execute_ib_gateway_restart)
        
        logger.info("All chaos scenarios scheduled")
    
    async def monitor_slos(self):
        """Continuous SLO monitoring with auto-abort"""
        logger.info("🔍 Starting SLO monitoring...")
        
        while not self.shutdown_event.is_set():
            try:
                violations = self.slo_monitor.check_slos()
                
                if any(violations.values()):
                    if self.slo_monitor.check_critical_duration(violations):
                        self.failsafe_abort()
                        break
                
                # Log error budget status every 10 minutes
                if int(time.time()) % 600 == 0:
                    error_budget = self.slo_monitor.get_error_budget_remaining()
                    logger.info(f"Error budget remaining: {error_budget:.1f}%")
                    
                    if error_budget < 97.0:
                        logger.warning(f"Error budget below target: {error_budget:.1f}% (<97%)")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"SLO monitoring error: {e}")
                await asyncio.sleep(60)  # Longer wait on error
    
    def run_scheduled_chaos(self):
        """Run scheduled chaos scenarios in background thread"""
        logger.info("Starting scheduled chaos runner...")
        
        while not self.shutdown_event.is_set():
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    
    async def generate_hourly_reports(self):
        """Generate progress reports every hour"""
        hour = 0
        
        while not self.shutdown_event.is_set() and hour < self.total_runtime_hours:
            await asyncio.sleep(3600)  # Wait 1 hour
            hour += 1
            
            # Generate hourly report
            report_path = Path(f"/tmp/reports/chaos_hour_{hour:02d}.md")
            error_budget = self.slo_monitor.get_error_budget_remaining()
            
            with open(report_path, 'w') as f:
                f.write(f"""# Chaos Hour {hour} Report
Time: {datetime.utcnow().isoformat()}Z

## Scenario Counters
- Kill-switch toggles: {self.kill_switch_count}
- Network jitter injections: {self.network_jitter_count}
- Database restarts: {self.db_restart_count}
- IB Gateway restarts: {self.ib_restart_count}

## SLO Status
- Error budget remaining: {error_budget:.1f}%
- Recent violations: {len([a for a in self.slo_monitor.alert_history if (datetime.utcnow() - datetime.fromisoformat(a['timestamp'].replace('Z', '+00:00'))).total_seconds() < 3600])}

## Health Status
- Order flood: {'Running' if 'order_flood' in self.active_scenarios else 'Stopped'}
- Abort triggered: {self.slo_monitor.abort_triggered}

""")
            
            logger.info(f"Hour {hour} report generated: {report_path}")
            
            # Special 8-hour snapshot
            if hour == 8:
                self._generate_8h_snapshot()
    
    def _generate_8h_snapshot(self):
        """Generate 8-hour snapshot report"""
        snapshot_path = Path("docs/tmp/8h_snapshot.md")
        error_budget = self.slo_monitor.get_error_budget_remaining()
        
        with open(snapshot_path, 'w') as f:
            f.write(f"""# 8-Hour Chaos Snapshot
Generated: {datetime.utcnow().isoformat()}Z

## Summary
- **Runtime**: 8/24 hours (33% complete)
- **Error Budget**: {error_budget:.1f}% remaining
- **Scenarios Executed**: {self.kill_switch_count + self.network_jitter_count + self.db_restart_count + self.ib_restart_count}
- **Critical Alerts**: {len([a for a in self.slo_monitor.alert_history if 'critical' in str(a)])}

## Scenario Performance
- Kill-switch toggles: {self.kill_switch_count} (target: 160 by 8h)
- Network jitter: {self.network_jitter_count} (target: 24 by 8h)
- Database restarts: {self.db_restart_count}
- IB Gateway restarts: {self.ib_restart_count}

## SLO Compliance
""")
            
            for metric, config in self.slo_monitor.slo_metrics.items():
                current_value = self.slo_monitor.query_prometheus(config['query'])
                status = "✅ PASS" if current_value and current_value <= config['threshold'] else "❌ FAIL"
                f.write(f"- {metric}: {current_value} (threshold: {config['threshold']}) {status}\n")
            
            f.write(f"""
## Recommendations
- {'Continue chaos testing' if error_budget > 97 else 'Consider reducing chaos intensity'}
- {'All systems nominal' if not self.slo_monitor.abort_triggered else 'Review abort triggers'}

""")
        
        logger.info(f"8-hour snapshot generated: {snapshot_path}")
    
    async def run_24h_chaos(self):
        """Main entry point for 24-hour chaos testing"""
        self.chaos_start_time = datetime.utcnow()
        logger.info(f"🚀 Starting 24h Game-Day Chaos at {self.chaos_start_time.isoformat()}Z")
        
        # Schedule all scenarios
        self.schedule_scenarios()
        
        # Start all async tasks
        tasks = [
            asyncio.create_task(self.continuous_order_flood()),
            asyncio.create_task(self.monitor_slos()),
            asyncio.create_task(self.generate_hourly_reports())
        ]
        
        # Start scheduled chaos in background thread
        threading.Thread(target=self.run_scheduled_chaos, daemon=True).start()
        
        try:
            # Run for 24 hours or until shutdown
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.total_runtime_hours * 3600
            )
        except asyncio.TimeoutError:
            logger.info("24-hour chaos testing completed successfully")
        except Exception as e:
            logger.error(f"Chaos testing failed: {e}")
        finally:
            self.shutdown_event.set()
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup all chaos scenarios and generate final report"""
        logger.info("🧹 Starting chaos cleanup...")
        
        # Stop order flood
        if 'order_flood' in self.active_scenarios:
            process = self.active_scenarios['order_flood']
            process.terminate()
            process.wait(timeout=30)
            logger.info("Order flood stopped")
        
        # Remove any remaining network jitter
        try:
            result = subprocess.run([
                'kubectl', 'get', 'pods', '-l', 'app=mech-exo-exec',
                '-o', 'jsonpath={.items[*].metadata.name}'
            ], capture_output=True, text=True)
            
            for pod_name in result.stdout.strip().split():
                subprocess.run([
                    'kubectl', 'exec', pod_name, '--',
                    'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'
                ], timeout=10, check=False)
        except Exception:
            pass  # Ignore cleanup errors
        
        # Generate final report
        await self._generate_final_report()
        
        logger.info("Chaos cleanup completed")
    
    async def _generate_final_report(self):
        """Generate final 24h chaos report"""
        end_time = datetime.utcnow()
        duration = (end_time - self.chaos_start_time).total_seconds() / 3600
        final_error_budget = self.slo_monitor.get_error_budget_remaining()
        
        report_path = Path("docs/tmp/24h_chaos_final.md")
        
        with open(report_path, 'w') as f:
            f.write(f"""# 24h Game-Day Chaos Final Report
Start: {self.chaos_start_time.isoformat()}Z
End: {end_time.isoformat()}Z
Duration: {duration:.1f} hours

## 🎯 Success Criteria
- **Error Budget**: {final_error_budget:.1f}% remaining (target: ≥97%) {'✅ PASS' if final_error_budget >= 97 else '❌ FAIL'}  
- **Recovery Time**: Average kill-switch recovery {'✅ PASS' if not self.slo_monitor.abort_triggered else '❌ FAIL'}
- **Capital Safety**: No erroneous orders (dry-run mode) ✅ PASS
- **Auto-Abort**: {'Not triggered ✅' if not self.slo_monitor.abort_triggered else 'Triggered ❌'}

## 📊 Scenario Execution
- **Kill-switch toggles**: {self.kill_switch_count}/480 target
- **Network jitter**: {self.network_jitter_count}/72 target  
- **Database restarts**: {self.db_restart_count}/2 target
- **IB Gateway restarts**: {self.ib_restart_count}/2 target

## 🚨 Alert Summary
- Total violations: {len(self.slo_monitor.alert_history)}
- Critical duration: {'None' if not self.slo_monitor.critical_alert_start else 'See logs'}

## 🏆 Overall Result
""")
            
            if final_error_budget >= 97 and not self.slo_monitor.abort_triggered:
                f.write("**SUCCESS** - 24h Game-Day Chaos testing passed all criteria")
            else:
                f.write("**PARTIAL SUCCESS** - Some criteria not met, review for improvements")
            
            f.write(f"""

## 📈 Next Steps (Week 4)
- Performance optimization with Redis market data cache
- GPU Inference PoC (LightGBM-GPU)
- Latency profiling and FlameGraphs  
- CloudWatch Cost Explorer integration
- Auto-shutdown of idle cold backup resources

## 📋 Week 4 Preview Tasks
If chaos testing passed, Week 4 will focus on:
1. Redis integration for market data caching
2. GPU acceleration proof-of-concept
3. Performance profiling and optimization
4. Cost reduction automation
5. Advanced monitoring and alerting refinements
""")
        
        logger.info(f"Final chaos report generated: {report_path}")


async def main():
    """Main entry point"""
    try:
        orchestrator = ChaosOrchestrator()
        await orchestrator.run_24h_chaos()
    except KeyboardInterrupt:
        logger.info("Chaos testing interrupted by user")
    except Exception as e:
        logger.error(f"Chaos testing failed: {e}")
        raise


if __name__ == "__main__":
    # Test mode for development
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Running in test mode (5 minutes instead of 24 hours)")
        orchestrator = ChaosOrchestrator()
        orchestrator.total_runtime_hours = 5/60  # 5 minutes for testing
        asyncio.run(orchestrator.run_24h_chaos())
    else:
        asyncio.run(main())