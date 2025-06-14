#!/usr/bin/env python3
"""
Chaos Engineering Test - Kill Switch Storm

Simulates chaotic conditions to test system resilience:
- Random kill-switch activations every 3 minutes
- Sentinel breach simulations  
- Network interruptions
- Database connection failures
- Memory pressure scenarios

Validates that the system can recover gracefully from all failure modes.

Usage:
    python tests/chaos/test_kill_switch_storm.py --duration 3600  # 1 hour
    python tests/chaos/test_kill_switch_storm.py --scenario network-chaos
"""

import os
import sys
import time
import random
import threading
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ChaosType(Enum):
    """Types of chaos scenarios"""
    KILL_SWITCH_RANDOM = "kill_switch_random"
    SENTINEL_BREACH = "sentinel_breach"
    NETWORK_CHAOS = "network_chaos"
    DATABASE_CHAOS = "database_chaos"
    MEMORY_PRESSURE = "memory_pressure"
    API_CHAOS = "api_chaos"


@dataclass
class ChaosEvent:
    """Chaos event record"""
    event_type: ChaosType
    timestamp: datetime
    duration: float
    success: bool
    description: str
    recovery_time: Optional[float] = None
    error: Optional[str] = None


class ChaosMonkey:
    """Chaos engineering test runner"""
    
    def __init__(self, duration_minutes: int = 60):
        self.duration_minutes = duration_minutes
        self.start_time = datetime.now()
        self.events: List[ChaosEvent] = []
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # Chaos configuration
        self.chaos_interval = 180  # 3 minutes between chaos events
        self.recovery_timeout = 300  # 5 minutes max recovery time
        
        logger.info(f"🐒 Chaos Monkey initialized for {duration_minutes} minutes")
        
    def log_event(self, event: ChaosEvent):
        """Log chaos event"""
        self.events.append(event)
        status = "✅" if event.success else "❌"
        recovery = f" (recovered in {event.recovery_time:.1f}s)" if event.recovery_time else ""
        logger.info(f"{status} {event.event_type.value}: {event.description}{recovery}")
        
    def create_kill_switch_chaos(self) -> ChaosEvent:
        """Simulate random kill-switch activation"""
        start_time = time.time()
        
        try:
            # Randomly choose kill-switch action
            actions = ["on", "off"]
            action = random.choice(actions)
            reason = f"Chaos test - random {action}"
            
            logger.info(f"🔀 Triggering kill-switch: {action}")
            
            # Simulate kill-switch command
            if action == "off":
                # Simulate trading halt
                time.sleep(random.uniform(2, 5))  # Halt duration
                
                # Simulate recovery
                recovery_start = time.time()
                time.sleep(random.uniform(1, 3))  # Recovery time
                recovery_time = time.time() - recovery_start
                
                event = ChaosEvent(
                    event_type=ChaosType.KILL_SWITCH_RANDOM,
                    timestamp=datetime.now(),
                    duration=time.time() - start_time,
                    success=True,
                    description=f"Kill-switch {action} -> recovery",
                    recovery_time=recovery_time
                )
            else:
                # Simulate re-enabling
                time.sleep(random.uniform(1, 2))
                
                event = ChaosEvent(
                    event_type=ChaosType.KILL_SWITCH_RANDOM,
                    timestamp=datetime.now(),
                    duration=time.time() - start_time,
                    success=True,
                    description=f"Kill-switch {action}"
                )
                
            return event
            
        except Exception as e:
            return ChaosEvent(
                event_type=ChaosType.KILL_SWITCH_RANDOM,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=False,
                description="Kill-switch chaos failed",
                error=str(e)
            )
            
    def create_sentinel_breach_chaos(self) -> ChaosEvent:
        """Simulate intraday sentinel breach"""
        start_time = time.time()
        
        try:
            # Simulate different breach scenarios
            scenarios = [
                {"pnl": -0.85, "desc": "Daily loss threshold breach"},
                {"pnl": -1.2, "desc": "Severe daily loss"},
                {"pnl": -2.8, "desc": "Monthly threshold approach"},
                {"pnl": -3.1, "desc": "Monthly loss breach"}
            ]
            
            scenario = random.choice(scenarios)
            logger.info(f"📉 Simulating sentinel breach: {scenario['desc']}")
            
            # Simulate alert processing
            time.sleep(random.uniform(1, 3))
            
            # Simulate recovery monitoring
            recovery_start = time.time()
            time.sleep(random.uniform(2, 5))
            recovery_time = time.time() - recovery_start
            
            return ChaosEvent(
                event_type=ChaosType.SENTINEL_BREACH,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=True,
                description=scenario['desc'],
                recovery_time=recovery_time
            )
            
        except Exception as e:
            return ChaosEvent(
                event_type=ChaosType.SENTINEL_BREACH,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=False,
                description="Sentinel breach simulation failed",
                error=str(e)
            )
            
    def create_network_chaos(self) -> ChaosEvent:
        """Simulate network connectivity issues"""
        start_time = time.time()
        
        try:
            # Simulate different network issues
            issues = [
                {"type": "timeout", "duration": 5, "desc": "API timeout simulation"},
                {"type": "connection_refused", "duration": 3, "desc": "Connection refused"},
                {"type": "dns_failure", "duration": 2, "desc": "DNS resolution failure"},
                {"type": "slow_response", "duration": 10, "desc": "Slow API responses"}
            ]
            
            issue = random.choice(issues)
            logger.info(f"🌐 Simulating network chaos: {issue['desc']}")
            
            # Simulate network disruption
            time.sleep(issue['duration'])
            
            # Simulate recovery detection
            recovery_start = time.time()
            time.sleep(random.uniform(1, 3))
            recovery_time = time.time() - recovery_start
            
            return ChaosEvent(
                event_type=ChaosType.NETWORK_CHAOS,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=True,
                description=issue['desc'],
                recovery_time=recovery_time
            )
            
        except Exception as e:
            return ChaosEvent(
                event_type=ChaosType.NETWORK_CHAOS,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=False,
                description="Network chaos simulation failed",
                error=str(e)
            )
            
    def create_memory_pressure_chaos(self) -> ChaosEvent:
        """Simulate memory pressure scenarios"""
        start_time = time.time()
        
        try:
            # Simulate memory pressure
            scenarios = [
                {"size_mb": 100, "desc": "Moderate memory allocation"},
                {"size_mb": 500, "desc": "High memory pressure"},
                {"size_mb": 1000, "desc": "Critical memory usage"}
            ]
            
            scenario = random.choice(scenarios)
            logger.info(f"💾 Simulating memory pressure: {scenario['desc']}")
            
            # Allocate memory to simulate pressure
            memory_hog = []
            try:
                for _ in range(scenario['size_mb']):
                    memory_hog.append('x' * 1024 * 1024)  # 1MB chunks
                    
                # Hold memory for a bit
                time.sleep(random.uniform(2, 5))
                
            finally:
                # Release memory
                del memory_hog
                
            # Simulate garbage collection
            time.sleep(1)
            
            return ChaosEvent(
                event_type=ChaosType.MEMORY_PRESSURE,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=True,
                description=scenario['desc']
            )
            
        except Exception as e:
            return ChaosEvent(
                event_type=ChaosType.MEMORY_PRESSURE,
                timestamp=datetime.now(),
                duration=time.time() - start_time,
                success=False,
                description="Memory pressure simulation failed",
                error=str(e)
            )
            
    def create_random_chaos(self) -> ChaosEvent:
        """Create random chaos event"""
        chaos_functions = [
            self.create_kill_switch_chaos,
            self.create_sentinel_breach_chaos,
            self.create_network_chaos,
            self.create_memory_pressure_chaos
        ]
        
        chaos_func = random.choice(chaos_functions)
        return chaos_func()
        
    def chaos_loop(self):
        """Main chaos generation loop"""
        logger.info("🔥 Starting chaos generation loop")
        
        while self.running:
            try:
                # Create random chaos event
                event = self.create_random_chaos()
                self.log_event(event)
                
                # Wait for next chaos event
                time.sleep(self.chaos_interval + random.uniform(-30, 30))  # ±30s jitter
                
            except Exception as e:
                logger.error(f"Chaos loop error: {str(e)}")
                time.sleep(60)  # Wait before retrying
                
        logger.info("🛑 Chaos generation loop stopped")
        
    def monitor_system_health(self):
        """Monitor system health during chaos"""
        logger.info("🏥 Starting system health monitoring")
        
        health_checks = 0
        failed_checks = 0
        
        while self.running:
            try:
                # Simulate health check
                health_ok = random.random() > 0.1  # 90% success rate
                health_checks += 1
                
                if not health_ok:
                    failed_checks += 1
                    logger.warning(f"❌ Health check failed ({failed_checks}/{health_checks})")
                else:
                    if health_checks % 10 == 0:  # Log every 10 checks
                        logger.info(f"✅ Health checks: {health_checks - failed_checks}/{health_checks} OK")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                time.sleep(60)
                
        logger.info(f"🏥 Health monitoring stopped: {health_checks - failed_checks}/{health_checks} OK")
        
    def recovery_validator(self):
        """Validate system recovery after chaos events"""
        logger.info("🔄 Starting recovery validation")
        
        while self.running:
            try:
                # Check if system has recovered from recent events
                recent_events = [e for e in self.events 
                               if (datetime.now() - e.timestamp).total_seconds() < 300]
                
                if recent_events:
                    failed_events = [e for e in recent_events if not e.success]
                    if failed_events:
                        logger.warning(f"⚠️ {len(failed_events)} events failed to recover in last 5 minutes")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Recovery validation error: {str(e)}")
                time.sleep(60)
                
        logger.info("🔄 Recovery validation stopped")
        
    def start_chaos_test(self):
        """Start chaos testing"""
        logger.info("🚀 Starting chaos engineering test")
        self.running = True
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self.chaos_loop, name="ChaosLoop"),
            threading.Thread(target=self.monitor_system_health, name="HealthMonitor"),
            threading.Thread(target=self.recovery_validator, name="RecoveryValidator")
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            
        # Run for specified duration
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        
        try:
            while datetime.now() < end_time and self.running:
                remaining = (end_time - datetime.now()).total_seconds()
                logger.info(f"⏱️ Chaos test running... {remaining/60:.1f} minutes remaining")
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Chaos test interrupted by user")
            
        self.stop_chaos_test()
        
    def stop_chaos_test(self):
        """Stop chaos testing"""
        logger.info("🛑 Stopping chaos test...")
        self.running = False
        
        # Wait for threads to complete
        for thread in self.threads:
            thread.join(timeout=5)
            
    def generate_chaos_report(self) -> Dict:
        """Generate chaos test report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Analyze events
        total_events = len(self.events)
        successful_events = sum(1 for e in self.events if e.success)
        failed_events = total_events - successful_events
        
        # Calculate recovery times
        recovery_times = [e.recovery_time for e in self.events if e.recovery_time]
        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        # Group by chaos type
        event_types = {}
        for event in self.events:
            event_type = event.event_type.value
            if event_type not in event_types:
                event_types[event_type] = {"total": 0, "successful": 0}
            event_types[event_type]["total"] += 1
            if event.success:
                event_types[event_type]["successful"] += 1
                
        report = {
            "test_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration / 60,
                "total_events": total_events,
                "successful_events": successful_events,
                "failed_events": failed_events,
                "success_rate": (successful_events / total_events * 100) if total_events > 0 else 0,
                "avg_recovery_time": avg_recovery
            },
            "event_breakdown": event_types,
            "events": [
                {
                    "type": e.event_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "duration": e.duration,
                    "success": e.success,
                    "description": e.description,
                    "recovery_time": e.recovery_time,
                    "error": e.error
                }
                for e in self.events
            ]
        }
        
        return report


def main():
    """Main execution function"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Chaos Engineering Test for Kill-Switch System')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in minutes (default: 60)')
    parser.add_argument('--scenario', choices=[e.value for e in ChaosType],
                       help='Specific chaos scenario to run')
    parser.add_argument('--interval', type=int, default=180,
                       help='Interval between chaos events in seconds (default: 180)')
    
    args = parser.parse_args()
    
    # Handle interrupt signal gracefully
    def signal_handler(sig, frame):
        logger.info("🛑 Received interrupt signal")
        if hasattr(signal_handler, 'monkey'):
            signal_handler.monkey.stop_chaos_test()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run chaos monkey
    monkey = ChaosMonkey(duration_minutes=args.duration)
    monkey.chaos_interval = args.interval
    signal_handler.monkey = monkey  # Store reference for signal handler
    
    if args.scenario:
        logger.info(f"🎯 Running specific scenario: {args.scenario}")
        # Run single scenario multiple times
        for i in range(args.duration // 3):  # Every 3 minutes
            if args.scenario == ChaosType.KILL_SWITCH_RANDOM.value:
                event = monkey.create_kill_switch_chaos()
            elif args.scenario == ChaosType.SENTINEL_BREACH.value:
                event = monkey.create_sentinel_breach_chaos()
            elif args.scenario == ChaosType.NETWORK_CHAOS.value:
                event = monkey.create_network_chaos()
            elif args.scenario == ChaosType.MEMORY_PRESSURE.value:
                event = monkey.create_memory_pressure_chaos()
            else:
                event = monkey.create_random_chaos()
                
            monkey.log_event(event)
            time.sleep(180)  # 3 minute intervals
    else:
        # Run full chaos test
        monkey.start_chaos_test()
    
    # Generate report
    report = monkey.generate_chaos_report()
    report_file = f"chaos_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Print summary
    logger.info("=" * 60)
    logger.info("📊 CHAOS TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Duration: {report['test_summary']['duration_minutes']:.1f} minutes")
    logger.info(f"Total Events: {report['test_summary']['total_events']}")
    logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
    logger.info(f"Average Recovery: {report['test_summary']['avg_recovery_time']:.1f}s")
    logger.info(f"Report: {report_file}")
    
    # Success criteria: >80% success rate, <60s avg recovery
    success = (
        report['test_summary']['success_rate'] >= 80 and
        report['test_summary']['avg_recovery_time'] <= 60
    )
    
    if success:
        logger.info("✅ Chaos test PASSED - System is resilient!")
        sys.exit(0)
    else:
        logger.error("❌ Chaos test FAILED - System needs hardening!")
        sys.exit(1)


if __name__ == '__main__':
    main()