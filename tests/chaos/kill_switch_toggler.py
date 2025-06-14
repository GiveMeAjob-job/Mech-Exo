#!/usr/bin/env python3
"""
Kill Switch Toggler - Chaos Monkey for Mech-Exo
Randomly toggles kill-switch every 3 minutes during load testing
"""

import asyncio
import aiohttp
import time
import logging
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [CHAOS] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChaosConfig:
    """Chaos testing configuration"""
    base_url: str = "http://localhost:8050"
    toggle_interval_min: int = 3  # minutes
    toggle_interval_max: int = 5  # minutes
    max_downtime: int = 60  # seconds
    test_duration: int = 3600  # seconds
    recovery_timeout: int = 120  # seconds


class KillSwitchToggler:
    """Chaos monkey for kill-switch testing"""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.start_time = None
        self.toggle_count = 0
        self.downtime_events = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_kill_switch_status(self) -> Optional[bool]:
        """Get current kill-switch status"""
        try:
            async with self.session.get(f"{self.config.base_url}/api/killswitch/status") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('enabled', False)
                else:
                    logger.warning(f"Failed to get kill-switch status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting kill-switch status: {str(e)}")
            return None
    
    async def toggle_kill_switch(self, enable: bool) -> bool:
        """Toggle kill-switch on/off"""
        try:
            payload = {
                'enabled': enable,
                'reason': f'chaos_test_{self.toggle_count}',
                'source': 'chaos_monkey'
            }
            
            async with self.session.post(
                f"{self.config.base_url}/api/killswitch/toggle",
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status == 200:
                    action = "ENABLED" if enable else "DISABLED"
                    logger.info(f"🔥 Kill-switch {action} (toggle #{self.toggle_count})")
                    return True
                else:
                    logger.error(f"Failed to toggle kill-switch: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error toggling kill-switch: {str(e)}")
            return False
    
    async def check_system_health(self) -> Dict[str, bool]:
        """Check system health endpoints"""
        health_status = {
            'healthz': False,
            'riskz': False,
            'trading_enabled': False
        }
        
        endpoints = [
            ('/healthz', 'healthz'),
            ('/riskz', 'riskz'),
            ('/api/status', 'trading_enabled')
        ]
        
        for endpoint, key in endpoints:
            try:
                async with self.session.get(f"{self.config.base_url}{endpoint}") as response:
                    if response.status == 200:
                        if key == 'trading_enabled':
                            data = await response.json()
                            health_status[key] = data.get('trading_enabled', False)
                        else:
                            health_status[key] = True
            except Exception as e:
                logger.debug(f"Health check failed for {endpoint}: {str(e)}")
        
        return health_status
    
    async def wait_for_recovery(self, timeout: int = 120) -> bool:
        """Wait for system to recover after kill-switch is disabled"""
        logger.info(f"⏳ Waiting for system recovery (timeout: {timeout}s)")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health = await self.check_system_health()
            
            if health['healthz'] and health['riskz'] and health['trading_enabled']:
                recovery_time = time.time() - start_time
                logger.info(f"✅ System recovered in {recovery_time:.1f}s")
                return True
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        logger.error(f"❌ System failed to recover within {timeout}s")
        return False
    
    async def chaos_cycle(self):
        """Execute one chaos cycle (enable -> wait -> disable -> wait for recovery)"""
        self.toggle_count += 1
        
        # Enable kill-switch
        if not await self.toggle_kill_switch(True):
            logger.error("Failed to enable kill-switch")
            return False
        
        # Record downtime start
        downtime_start = time.time()
        
        # Wait for configured downtime
        downtime_duration = random.randint(10, self.config.max_downtime)
        logger.info(f"💤 Kill-switch active for {downtime_duration}s")
        await asyncio.sleep(downtime_duration)
        
        # Disable kill-switch
        if not await self.toggle_kill_switch(False):
            logger.error("Failed to disable kill-switch")
            return False
        
        # Wait for recovery
        recovery_success = await self.wait_for_recovery(self.config.recovery_timeout)
        
        # Record event
        total_downtime = time.time() - downtime_start
        event = {
            'toggle_count': self.toggle_count,
            'downtime_start': datetime.fromtimestamp(downtime_start).isoformat(),
            'planned_downtime': downtime_duration,
            'actual_downtime': total_downtime,
            'recovery_success': recovery_success,
            'recovery_time': total_downtime - downtime_duration if recovery_success else None
        }
        
        self.downtime_events.append(event)
        
        logger.info(
            f"📊 Chaos cycle #{self.toggle_count} completed: "
            f"downtime={total_downtime:.1f}s, recovery={'OK' if recovery_success else 'FAILED'}"
        )
        
        return recovery_success
    
    async def run_chaos_test(self) -> Dict:
        """Run the complete chaos test"""
        logger.info("🔥 Starting kill-switch chaos testing")
        logger.info(f"Duration: {self.config.test_duration}s")
        logger.info(f"Toggle interval: {self.config.toggle_interval_min}-{self.config.toggle_interval_max} minutes")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            while self.running:
                elapsed_time = time.time() - self.start_time
                
                # Check if test duration exceeded
                if elapsed_time > self.config.test_duration:
                    logger.info("Test duration exceeded, stopping chaos test")
                    break
                
                # Execute chaos cycle
                success = await self.chaos_cycle()
                
                if not success:
                    logger.error("Chaos cycle failed, continuing anyway...")
                
                # Wait for next toggle (random interval)
                wait_minutes = random.randint(
                    self.config.toggle_interval_min,
                    self.config.toggle_interval_max
                )
                wait_seconds = wait_minutes * 60
                
                logger.info(f"⏰ Next chaos cycle in {wait_minutes} minutes")
                
                # Wait with ability to stop early
                for _ in range(wait_seconds):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Chaos test interrupted by user")
        finally:
            # Ensure kill-switch is disabled
            current_status = await self.get_kill_switch_status()
            if current_status:
                logger.info("🧹 Disabling kill-switch before exit")
                await self.toggle_kill_switch(False)
                await self.wait_for_recovery()
        
        # Generate report
        test_duration = time.time() - self.start_time
        report = {
            'test_duration': test_duration,
            'total_toggles': self.toggle_count,
            'downtime_events': self.downtime_events,
            'successful_recoveries': sum(1 for e in self.downtime_events if e['recovery_success']),
            'config': {
                'toggle_interval_min': self.config.toggle_interval_min,
                'toggle_interval_max': self.config.toggle_interval_max,
                'max_downtime': self.config.max_downtime,
                'recovery_timeout': self.config.recovery_timeout
            }
        }
        
        logger.info("🔥 Chaos test completed")
        logger.info(f"Total duration: {test_duration:.1f}s")
        logger.info(f"Total toggles: {self.toggle_count}")
        logger.info(f"Successful recoveries: {report['successful_recoveries']}/{self.toggle_count}")
        
        return report


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Kill-switch chaos testing')
    parser.add_argument('--url', default='http://localhost:8050',
                       help='Base URL for the API')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Test duration (seconds)')
    parser.add_argument('--min-interval', type=int, default=3,
                       help='Minimum toggle interval (minutes)')
    parser.add_argument('--max-interval', type=int, default=5,
                       help='Maximum toggle interval (minutes)')
    parser.add_argument('--max-downtime', type=int, default=60,
                       help='Maximum downtime per toggle (seconds)')
    
    args = parser.parse_args()
    
    config = ChaosConfig(
        base_url=args.url,
        test_duration=args.duration,
        toggle_interval_min=args.min_interval,
        toggle_interval_max=args.max_interval,
        max_downtime=args.max_downtime
    )
    
    # Run chaos test
    async with KillSwitchToggler(config) as toggler:
        try:
            report = await toggler.run_chaos_test()
            
            # Save report
            report_file = f"chaos_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Chaos test report saved to {report_file}")
            
            # Return success if all recoveries succeeded
            success_rate = report['successful_recoveries'] / max(report['total_toggles'], 1)
            if success_rate >= 0.95:  # 95% success rate threshold
                logger.info("Chaos test passed")
                return 0
            else:
                logger.error(f"Chaos test failed - success rate: {success_rate:.1%}")
                return 1
                
        except Exception as e:
            logger.error(f"Chaos test failed with exception: {str(e)}")
            return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))