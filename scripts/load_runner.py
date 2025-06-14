#!/usr/bin/env python3
"""
Load Generator for Chaos Testing
Generates sustained load at configurable rates with adaptive backoff
"""

import asyncio
import aiohttp
import argparse
import logging
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Optional
import signal
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('load_runner')


class LoadGenerator:
    """Generates HTTP load with rate limiting and adaptive backoff"""
    
    def __init__(self, endpoint: str, rate: int, duration: int, dry_run: bool = True):
        self.endpoint = endpoint
        self.target_rate = rate  # requests per second
        self.duration = duration  # seconds
        self.dry_run = dry_run
        
        # Statistics
        self.requests_sent = 0
        self.requests_successful = 0
        self.requests_failed = 0
        self.rate_limited = 0
        self.start_time = None
        self.end_time = None
        
        # Rate limiting
        self.current_rate = rate
        self.backoff_factor = 0.8
        self.recovery_factor = 1.1
        self.min_rate = 5
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
    
    def _generate_test_order(self) -> Dict:
        """Generate a realistic test order payload"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        sides = ['BUY', 'SELL']
        
        return {
            'symbol': random.choice(symbols),
            'side': random.choice(sides),
            'quantity': random.randint(10, 1000),
            'order_type': 'MARKET',
            'dry_run': self.dry_run,
            'chaos_test': True,
            'timestamp': datetime.utcnow().isoformat(),
            'test_id': f'chaos_{int(time.time())}_{random.randint(1000, 9999)}'
        }
    
    async def _send_request(self, session: aiohttp.ClientSession) -> bool:
        """Send a single HTTP request"""
        payload = self._generate_test_order()
        
        try:
            async with session.post(
                self.endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                self.requests_sent += 1
                
                if response.status == 200:
                    self.requests_successful += 1
                    return True
                elif response.status == 429:  # Rate limited
                    self.rate_limited += 1
                    logger.debug(f"Rate limited (429) - reducing rate")
                    self._reduce_rate()
                    return False
                else:
                    self.requests_failed += 1
                    logger.debug(f"Request failed with status {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            self.requests_failed += 1
            logger.debug("Request timed out")
            return False
        except Exception as e:
            self.requests_failed += 1
            logger.debug(f"Request error: {e}")
            return False
    
    def _reduce_rate(self):
        """Reduce current rate due to rate limiting"""
        new_rate = max(self.min_rate, int(self.current_rate * self.backoff_factor))
        if new_rate != self.current_rate:
            logger.info(f"Rate reduced: {self.current_rate} -> {new_rate} req/s")
            self.current_rate = new_rate
    
    def _increase_rate(self):
        """Gradually increase rate if performing well"""
        new_rate = min(self.target_rate, int(self.current_rate * self.recovery_factor))
        if new_rate != self.current_rate:
            logger.info(f"Rate increased: {self.current_rate} -> {new_rate} req/s")
            self.current_rate = new_rate
    
    def _log_stats(self):
        """Log current statistics"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            actual_rate = self.requests_sent / elapsed if elapsed > 0 else 0
            success_rate = (self.requests_successful / self.requests_sent * 100) if self.requests_sent > 0 else 0
            
            logger.info(
                f"Stats: {self.requests_sent} sent, {self.requests_successful} OK, "
                f"{self.requests_failed} failed, {self.rate_limited} rate-limited, "
                f"Rate: {actual_rate:.1f} req/s, Success: {success_rate:.1f}%"
            )
    
    async def run(self):
        """Run the load generator"""
        self.start_time = time.time()
        logger.info(f"Starting load generation: {self.current_rate} req/s for {self.duration}s")
        logger.info(f"Target endpoint: {self.endpoint}")
        logger.info(f"Dry run mode: {self.dry_run}")
        
        # Calculate timing
        interval = 1.0 / self.current_rate
        end_time = self.start_time + self.duration
        last_stats_time = self.start_time
        last_rate_adjustment = self.start_time
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while time.time() < end_time and not self.shutdown_event.is_set():
                request_start = time.time()
                
                # Send request
                success = await self._send_request(session)
                
                # Adaptive rate adjustment
                current_time = time.time()
                if current_time - last_rate_adjustment > 30:  # Adjust every 30 seconds
                    if success and self.current_rate < self.target_rate:
                        # Gradually increase rate if doing well
                        recent_success_rate = self.requests_successful / max(1, self.requests_sent)
                        if recent_success_rate > 0.95:  # 95% success rate
                            self._increase_rate()
                    last_rate_adjustment = current_time
                
                # Log stats every 60 seconds
                if current_time - last_stats_time > 60:
                    self._log_stats()
                    last_stats_time = current_time
                
                # Rate limiting - wait for next request
                request_duration = time.time() - request_start
                sleep_time = max(0, (1.0 / self.current_rate) - request_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        self.end_time = time.time()
        self._log_final_stats()
    
    def _log_final_stats(self):
        """Log final statistics"""
        total_duration = self.end_time - self.start_time
        actual_rate = self.requests_sent / total_duration if total_duration > 0 else 0
        success_rate = (self.requests_successful / self.requests_sent * 100) if self.requests_sent > 0 else 0
        
        logger.info("=" * 60)
        logger.info("FINAL LOAD GENERATION STATS")
        logger.info("=" * 60)
        logger.info(f"Duration: {total_duration:.1f}s")
        logger.info(f"Target rate: {self.target_rate} req/s")
        logger.info(f"Actual rate: {actual_rate:.1f} req/s")
        logger.info(f"Total requests: {self.requests_sent}")
        logger.info(f"Successful: {self.requests_successful} ({success_rate:.1f}%)")
        logger.info(f"Failed: {self.requests_failed}")
        logger.info(f"Rate limited: {self.rate_limited}")
        logger.info("=" * 60)
        
        # Write stats to file for analysis
        stats = {
            'duration': total_duration,
            'target_rate': self.target_rate,
            'actual_rate': actual_rate,
            'requests_sent': self.requests_sent,
            'requests_successful': self.requests_successful,
            'requests_failed': self.requests_failed,
            'rate_limited': self.rate_limited,
            'success_rate': success_rate,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open('/tmp/chaos_logs/load_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Load generation statistics saved to /tmp/chaos_logs/load_stats.json")


def main():
    parser = argparse.ArgumentParser(description='HTTP Load Generator for Chaos Testing')
    parser.add_argument('--rate', type=int, default=50, help='Target requests per second')
    parser.add_argument('--duration', type=int, default=3600, help='Duration in seconds')
    parser.add_argument('--endpoint', type=str, default='http://localhost:8000/api/orders', help='Target endpoint')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Enable dry-run mode (default: True)')
    parser.add_argument('--live', action='store_true', help='Disable dry-run mode (enable live orders)')
    
    args = parser.parse_args()
    
    # Safety check for live mode
    if args.live:
        dry_run = False
        print("⚠️  WARNING: Live mode enabled - real orders will be sent!")
        print("Press Ctrl+C within 10 seconds to cancel...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("Load generation cancelled")
            sys.exit(0)
    else:
        dry_run = True
    
    # Create and run load generator
    generator = LoadGenerator(
        endpoint=args.endpoint,
        rate=args.rate,
        duration=args.duration,
        dry_run=dry_run
    )
    
    try:
        asyncio.run(generator.run())
    except KeyboardInterrupt:
        logger.info("Load generation stopped by user")
    except Exception as e:
        logger.error(f"Load generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()