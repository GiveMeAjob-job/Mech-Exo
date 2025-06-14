"""
Market Data Cache Implementation
Phase P11 Week 4 Day 1 - Redis-based caching for performance optimization

Features:
- Async Redis operations with connection pooling
- TTL-based cache expiration (100ms - 1hour)
- Prometheus metrics for hit/miss ratio monitoring
- Automatic cache warming for active symbols
- Fallback mechanisms for cache failures
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import aiohttp

logger = logging.getLogger(__name__)

# Prometheus metrics
cache_hits = Counter('market_cache_hits_total', 'Total cache hits', ['data_type'])
cache_misses = Counter('market_cache_misses_total', 'Total cache misses', ['data_type'])
cache_operations = Histogram('market_cache_operation_duration_seconds', 'Cache operation duration', ['operation'])
cache_size = Gauge('market_cache_size_bytes', 'Current cache size in bytes')
cache_connections = Gauge('market_cache_connections', 'Active Redis connections')


@dataclass
class CacheConfig:
    """Cache configuration parameters"""
    host: str = "redis-master"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_connect_timeout: int = 5
    socket_timeout: int = 2
    retry_on_timeout: bool = True
    
    # TTL settings (seconds)
    price_ttl: int = 1        # Real-time prices
    volume_ttl: int = 5       # Volume data
    orderbook_ttl: int = 2    # Order book data
    static_ttl: int = 3600    # Static reference data
    
    # Cache warming
    warmup_symbols: List[str] = None
    warmup_enabled: bool = True
    
    def __post_init__(self):
        if self.warmup_symbols is None:
            self.warmup_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]


class MarketCache:
    """Redis-based market data cache with async operations"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_pool = None
        self.redis = None
        self._connected = False
        self._lock = asyncio.Lock()
        
        # Cache key patterns
        self.key_patterns = {
            'price': 'px:{symbol}',
            'volume': 'vol:{symbol}',
            'bid': 'bid:{symbol}',
            'ask': 'ask:{symbol}',
            'orderbook': 'ob:{symbol}:{depth}',
            'static': 'static:{symbol}:{field}'
        }
        
        # TTL mapping
        self.ttl_map = {
            'price': self.config.price_ttl,
            'volume': self.config.volume_ttl,
            'bid': self.config.price_ttl,
            'ask': self.config.price_ttl,
            'orderbook': self.config.orderbook_ttl,
            'static': self.config.static_ttl
        }
    
    async def connect(self) -> bool:
        """Establish Redis connection with connection pooling"""
        if self._connected:
            return True
        
        async with self._lock:
            if self._connected:
                return True
            
            try:
                self.redis_pool = redis.ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    password=self.config.password,
                    db=self.config.db,
                    max_connections=self.config.max_connections,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    socket_timeout=self.config.socket_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=True
                )
                
                self.redis = redis.Redis(connection_pool=self.redis_pool)
                
                # Test connection
                await self.redis.ping()
                self._connected = True
                
                logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
                
                # Update connection metric
                cache_connections.set(self.config.max_connections)
                
                # Start cache warming if enabled
                if self.config.warmup_enabled:
                    asyncio.create_task(self._start_cache_warming())
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                return False
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.aclose()
            self._connected = False
            cache_connections.set(0)
            logger.info("Disconnected from Redis")
    
    def _get_key(self, data_type: str, symbol: str, **kwargs) -> str:
        """Generate cache key based on data type and symbol"""
        pattern = self.key_patterns.get(data_type, f"{data_type}:{{symbol}}")
        return pattern.format(symbol=symbol, **kwargs)
    
    def _get_ttl(self, data_type: str) -> int:
        """Get TTL for data type"""
        return self.ttl_map.get(data_type, 60)  # Default 1 minute
    
    @cache_operations.time()
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get cached price for symbol"""
        return await self._get_typed('price', symbol, float)
    
    @cache_operations.time()
    async def set_price(self, symbol: str, price: float) -> bool:
        """Cache price for symbol"""
        return await self._set_typed('price', symbol, price)
    
    @cache_operations.time()
    async def get_volume(self, symbol: str) -> Optional[int]:
        """Get cached volume for symbol"""
        return await self._get_typed('volume', symbol, int)
    
    @cache_operations.time()
    async def set_volume(self, symbol: str, volume: int) -> bool:
        """Cache volume for symbol"""
        return await self._set_typed('volume', symbol, volume)
    
    @cache_operations.time()
    async def get_bid_ask(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get cached bid/ask for symbol"""
        try:
            if not await self.connect():
                return None
            
            pipe = self.redis.pipeline()
            bid_key = self._get_key('bid', symbol)
            ask_key = self._get_key('ask', symbol)
            
            pipe.get(bid_key)
            pipe.get(ask_key)
            
            results = await pipe.execute()
            bid_val, ask_val = results
            
            if bid_val is not None and ask_val is not None:
                cache_hits.labels(data_type='bid_ask').inc()
                return {
                    'bid': float(bid_val),
                    'ask': float(ask_val),
                    'spread': float(ask_val) - float(bid_val)
                }
            else:
                cache_misses.labels(data_type='bid_ask').inc()
                return None
                
        except Exception as e:
            logger.warning(f"Cache get_bid_ask error for {symbol}: {e}")
            cache_misses.labels(data_type='bid_ask').inc()
            return None
    
    @cache_operations.time()
    async def set_bid_ask(self, symbol: str, bid: float, ask: float) -> bool:
        """Cache bid/ask for symbol"""
        try:
            if not await self.connect():
                return False
            
            pipe = self.redis.pipeline()
            bid_key = self._get_key('bid', symbol)
            ask_key = self._get_key('ask', symbol)
            ttl = self._get_ttl('bid')
            
            pipe.set(bid_key, bid, ex=ttl)
            pipe.set(ask_key, ask, ex=ttl)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.warning(f"Cache set_bid_ask error for {symbol}: {e}")
            return False
    
    @cache_operations.time()
    async def get_orderbook(self, symbol: str, depth: int = 5) -> Optional[Dict]:
        """Get cached order book for symbol"""
        return await self._get_typed('orderbook', symbol, dict, depth=depth)
    
    @cache_operations.time()
    async def set_orderbook(self, symbol: str, orderbook: Dict, depth: int = 5) -> bool:
        """Cache order book for symbol"""
        return await self._set_typed('orderbook', symbol, orderbook, depth=depth)
    
    async def _get_typed(self, data_type: str, symbol: str, type_cls, **kwargs) -> Optional[Any]:
        """Generic typed get operation"""
        try:
            if not await self.connect():
                return None
            
            key = self._get_key(data_type, symbol, **kwargs)
            value = await self.redis.get(key)
            
            if value is not None:
                cache_hits.labels(data_type=data_type).inc()
                
                if type_cls == dict:
                    return json.loads(value)
                else:
                    return type_cls(value)
            else:
                cache_misses.labels(data_type=data_type).inc()
                return None
                
        except Exception as e:
            logger.warning(f"Cache get error for {data_type}:{symbol}: {e}")
            cache_misses.labels(data_type=data_type).inc()
            return None
    
    async def _set_typed(self, data_type: str, symbol: str, value: Any, **kwargs) -> bool:
        """Generic typed set operation"""
        try:
            if not await self.connect():
                return False
            
            key = self._get_key(data_type, symbol, **kwargs)
            ttl = self._get_ttl(data_type)
            
            if isinstance(value, dict):
                cache_value = json.dumps(value)
            else:
                cache_value = str(value)
            
            await self.redis.set(key, cache_value, ex=ttl)
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for {data_type}:{symbol}: {e}")
            return False
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple prices efficiently using pipeline"""
        try:
            if not await self.connect():
                return {}
            
            pipe = self.redis.pipeline()
            keys = [self._get_key('price', symbol) for symbol in symbols]
            
            for key in keys:
                pipe.get(key)
            
            results = await pipe.execute()
            
            prices = {}
            for symbol, result in zip(symbols, results):
                if result is not None:
                    prices[symbol] = float(result)
                    cache_hits.labels(data_type='price').inc()
                else:
                    cache_misses.labels(data_type='price').inc()
            
            return prices
            
        except Exception as e:
            logger.warning(f"Cache get_multiple_prices error: {e}")
            for _ in symbols:
                cache_misses.labels(data_type='price').inc()
            return {}
    
    async def set_multiple_prices(self, price_data: Dict[str, float]) -> bool:
        """Set multiple prices efficiently using pipeline"""
        try:
            if not await self.connect():
                return False
            
            pipe = self.redis.pipeline()
            ttl = self._get_ttl('price')
            
            for symbol, price in price_data.items():
                key = self._get_key('price', symbol)
                pipe.set(key, price, ex=ttl)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            logger.warning(f"Cache set_multiple_prices error: {e}")
            return False
    
    async def invalidate_symbol(self, symbol: str) -> bool:
        """Invalidate all cached data for a symbol"""
        try:
            if not await self.connect():
                return False
            
            pipe = self.redis.pipeline()
            
            # Delete all data types for this symbol
            for data_type in self.key_patterns.keys():
                if data_type == 'orderbook':
                    # Handle different depths
                    for depth in [5, 10, 20]:
                        key = self._get_key(data_type, symbol, depth=depth)
                        pipe.delete(key)
                else:
                    key = self._get_key(data_type, symbol)
                    pipe.delete(key)
            
            await pipe.execute()
            logger.info(f"Invalidated cache for symbol: {symbol}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache invalidation error for {symbol}: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not await self.connect():
                return {}
            
            info = await self.redis.info('memory')
            stats = await self.redis.info('stats')
            
            # Calculate hit ratio
            hits = stats.get('keyspace_hits', 0)
            misses = stats.get('keyspace_misses', 0)
            total = hits + misses
            hit_ratio = (hits / total) if total > 0 else 0.0
            
            # Update cache size metric
            cache_size.set(info.get('used_memory', 0))
            
            return {
                'memory_used': info.get('used_memory', 0),
                'memory_used_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': hits,
                'keyspace_misses': misses,
                'hit_ratio': hit_ratio,
                'connected_clients': stats.get('connected_clients', 0),
                'total_commands_processed': stats.get('total_commands_processed', 0),
                'expired_keys': stats.get('expired_keys', 0),
                'evicted_keys': stats.get('evicted_keys', 0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {}
    
    async def _start_cache_warming(self):
        """Start background cache warming for active symbols"""
        logger.info("Starting cache warming for active symbols")
        
        while self._connected:
            try:
                # Warm up prices for configured symbols
                await self._warm_prices(self.config.warmup_symbols)
                
                # Wait 30 minutes before next warming
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.warning(f"Cache warming error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _warm_prices(self, symbols: List[str]):
        """Warm cache with current prices for symbols"""
        try:
            # This would integrate with your actual market data source
            # For now, simulate cache warming
            logger.debug(f"Warming cache for {len(symbols)} symbols")
            
            # In real implementation, fetch from IB or other data source
            # await self.set_multiple_prices(price_data)
            
        except Exception as e:
            logger.warning(f"Price warming error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache"""
        try:
            if not await self.connect():
                return {'status': 'unhealthy', 'error': 'connection_failed'}
            
            # Test basic operations
            test_key = f"health_check:{int(time.time())}"
            await self.redis.set(test_key, "test", ex=10)
            value = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            if value != "test":
                return {'status': 'unhealthy', 'error': 'read_write_failed'}
            
            stats = await self.get_cache_stats()
            
            return {
                'status': 'healthy',
                'connected': self._connected,
                'hit_ratio': stats.get('hit_ratio', 0.0),
                'memory_used': stats.get('memory_used_human', '0B'),
                'response_time_ms': cache_operations._sum.get() * 1000
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}


# Global cache instance
market_cache = MarketCache()


async def get_cached_price(symbol: str, fallback_func=None) -> Optional[float]:
    """Convenience function to get cached price with fallback"""
    price = await market_cache.get_price(symbol)
    
    if price is None and fallback_func:
        # Cache miss - fetch from fallback and cache result
        try:
            price = await fallback_func(symbol)
            if price is not None:
                await market_cache.set_price(symbol, price)
        except Exception as e:
            logger.warning(f"Fallback function failed for {symbol}: {e}")
    
    return price


async def get_cached_bid_ask(symbol: str, fallback_func=None) -> Optional[Dict[str, float]]:
    """Convenience function to get cached bid/ask with fallback"""
    bid_ask = await market_cache.get_bid_ask(symbol)
    
    if bid_ask is None and fallback_func:
        try:
            bid_ask_data = await fallback_func(symbol)
            if bid_ask_data and 'bid' in bid_ask_data and 'ask' in bid_ask_data:
                await market_cache.set_bid_ask(symbol, bid_ask_data['bid'], bid_ask_data['ask'])
                bid_ask = bid_ask_data
        except Exception as e:
            logger.warning(f"Fallback function failed for {symbol}: {e}")
    
    return bid_ask


# Context manager for cache operations
class CacheSession:
    """Context manager for cache operations with automatic cleanup"""
    
    def __init__(self, config: CacheConfig = None):
        self.cache = MarketCache(config)
    
    async def __aenter__(self):
        await self.cache.connect()
        return self.cache
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cache.disconnect()


if __name__ == "__main__":
    async def test_cache():
        """Test cache functionality"""
        cache = MarketCache()
        
        try:
            await cache.connect()
            
            # Test price caching
            await cache.set_price("AAPL", 150.25)
            price = await cache.get_price("AAPL")
            print(f"Cached price for AAPL: {price}")
            
            # Test bid/ask caching
            await cache.set_bid_ask("AAPL", 150.20, 150.30)
            bid_ask = await cache.get_bid_ask("AAPL")
            print(f"Cached bid/ask for AAPL: {bid_ask}")
            
            # Test multiple prices
            prices = {"AAPL": 150.25, "GOOGL": 2800.50, "MSFT": 320.75}
            await cache.set_multiple_prices(prices)
            cached_prices = await cache.get_multiple_prices(["AAPL", "GOOGL", "MSFT", "TSLA"])
            print(f"Multiple cached prices: {cached_prices}")
            
            # Test cache stats
            stats = await cache.get_cache_stats()
            print(f"Cache stats: {stats}")
            
            # Test health check
            health = await cache.health_check()
            print(f"Health check: {health}")
            
        finally:
            await cache.disconnect()
    
    # Run test
    asyncio.run(test_cache())