"""
Circuit Breaker Pattern Implementation
Phase P11 Week 4 Day 5 - Resilient service communication with fallback

Features:
- Circuit breaker with configurable thresholds
- Exponential backoff retry mechanism
- Half-open state for gradual recovery
- Prometheus metrics integration
- Multiple circuit breaker strategies
- Async and sync support
"""

import time
import logging
import asyncio
import functools
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import inspect

from prometheus_client import Counter, Histogram, Gauge, Enum as PrometheusEnum

logger = logging.getLogger(__name__)

# Prometheus metrics
circuit_state_changes = Counter('circuit_breaker_state_changes_total', 'Circuit breaker state changes', ['circuit_name', 'from_state', 'to_state'])
circuit_requests = Counter('circuit_breaker_requests_total', 'Circuit breaker requests', ['circuit_name', 'status'])
circuit_failures = Counter('circuit_breaker_failures_total', 'Circuit breaker failures', ['circuit_name', 'error_type'])
circuit_fallbacks = Counter('circuit_breaker_fallbacks_total', 'Circuit breaker fallback executions', ['circuit_name'])
circuit_response_time = Histogram('circuit_breaker_response_time_seconds', 'Circuit breaker response time', ['circuit_name'])
circuit_current_state = PrometheusEnum('circuit_breaker_current_state', 'Current circuit breaker state', ['circuit_name'], states=['closed', 'open', 'half_open'])


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit is open, failing fast
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    # Failure threshold
    failure_threshold: int = 5          # Number of failures to open circuit
    failure_rate_threshold: float = 0.5 # Percentage of failures to open circuit
    
    # Time windows
    timeout_duration: float = 60.0      # How long circuit stays open (seconds)
    rolling_window: float = 60.0        # Rolling window for failure calculation (seconds)
    
    # Half-open testing
    half_open_max_calls: int = 3        # Max calls in half-open state
    
    # Retry configuration
    enable_retry: bool = True
    max_retry_attempts: int = 3
    base_delay: float = 1.0             # Base delay for exponential backoff
    max_delay: float = 60.0             # Maximum delay
    jitter: bool = True                 # Add jitter to retry delays
    
    # Monitoring
    enable_metrics: bool = True
    name: str = "default"


@dataclass
class CallRecord:
    """Record of a function call"""
    timestamp: float
    success: bool
    response_time: float
    error_type: Optional[str] = None


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, circuit_name: str, message: str = None):
        self.circuit_name = circuit_name
        self.message = message or f"Circuit breaker '{circuit_name}' is open"
        super().__init__(self.message)


class CircuitBreaker:
    """Circuit breaker implementation with metrics and fallback support"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.call_history: List[CallRecord] = []
        self._lock = threading.RLock()
        
        # State change tracking
        self.state_change_time = time.time()
        
        # Initialize metrics
        if self.config.enable_metrics:
            circuit_current_state.labels(circuit_name=self.config.name).state(self.state.value)
    
    def _record_call(self, success: bool, response_time: float, error_type: str = None):
        """Record a function call result"""
        with self._lock:
            # Clean old records outside rolling window
            current_time = time.time()
            cutoff_time = current_time - self.config.rolling_window
            self.call_history = [record for record in self.call_history if record.timestamp > cutoff_time]
            
            # Add new record
            record = CallRecord(
                timestamp=current_time,
                success=success,
                response_time=response_time,
                error_type=error_type
            )
            self.call_history.append(record)
            
            # Update metrics
            if self.config.enable_metrics:
                status = "success" if success else "failure"
                circuit_requests.labels(circuit_name=self.config.name, status=status).inc()
                circuit_response_time.labels(circuit_name=self.config.name).observe(response_time)
                
                if not success and error_type:
                    circuit_failures.labels(circuit_name=self.config.name, error_type=error_type).inc()
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate within rolling window"""
        if not self.call_history:
            return 0.0
        
        total_calls = len(self.call_history)
        failed_calls = sum(1 for record in self.call_history if not record.success)
        
        return failed_calls / total_calls if total_calls > 0 else 0.0
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        failure_rate = self._calculate_failure_rate()
        
        # Check both absolute count and rate thresholds
        return (
            len([r for r in self.call_history if not r.success]) >= self.config.failure_threshold or
            failure_rate >= self.config.failure_rate_threshold
        )
    
    def _transition_state(self, new_state: CircuitState):
        """Transition to a new state"""
        if new_state == self.state:
            return
        
        old_state = self.state
        self.state = new_state
        self.state_change_time = time.time()
        
        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0
        
        # Update metrics
        if self.config.enable_metrics:
            circuit_state_changes.labels(
                circuit_name=self.config.name,
                from_state=old_state.value,
                to_state=new_state.value
            ).inc()
            circuit_current_state.labels(circuit_name=self.config.name).state(new_state.value)
        
        logger.info(f"Circuit breaker '{self.config.name}' transitioned from {old_state.value} to {new_state.value}")
    
    def _can_execute(self) -> bool:
        """Check if a call can be executed based on current state"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if current_time - self.state_change_time >= self.config.timeout_duration:
                self._transition_state(CircuitState.HALF_OPEN)
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited calls to test recovery
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def _handle_success(self, response_time: float):
        """Handle successful call"""
        with self._lock:
            self._record_call(True, response_time)
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                # If enough successful calls in half-open, close circuit
                if self.half_open_calls >= self.config.half_open_max_calls:
                    self._transition_state(CircuitState.CLOSED)
    
    def _handle_failure(self, response_time: float, error: Exception):
        """Handle failed call"""
        with self._lock:
            error_type = type(error).__name__
            self._record_call(False, response_time, error_type)
            
            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens circuit
                self._transition_state(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                # Check if we should open circuit
                if self._should_open_circuit():
                    self._transition_state(CircuitState.OPEN)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self._can_execute():
            raise CircuitBreakerError(self.config.name)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            self._handle_success(response_time)
            return result
        
        except Exception as e:
            response_time = time.time() - start_time
            self._handle_failure(response_time, e)
            raise
    
    async def call_async(self, coro: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if not self._can_execute():
            raise CircuitBreakerError(self.config.name)
        
        start_time = time.time()
        
        try:
            if inspect.iscoroutinefunction(coro):
                result = await coro(*args, **kwargs)
            else:
                result = coro(*args, **kwargs)
                
            response_time = time.time() - start_time
            self._handle_success(response_time)
            return result
        
        except Exception as e:
            response_time = time.time() - start_time
            self._handle_failure(response_time, e)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics"""
        with self._lock:
            total_calls = len(self.call_history)
            failed_calls = sum(1 for record in self.call_history if not record.success)
            
            if total_calls > 0:
                avg_response_time = sum(record.response_time for record in self.call_history) / total_calls
                failure_rate = failed_calls / total_calls
            else:
                avg_response_time = 0.0
                failure_rate = 0.0
            
            return {
                'name': self.config.name,
                'state': self.state.value,
                'total_calls': total_calls,
                'failed_calls': failed_calls,
                'failure_rate': failure_rate,
                'avg_response_time': avg_response_time,
                'state_change_time': self.state_change_time,
                'half_open_calls': self.half_open_calls if self.state == CircuitState.HALF_OPEN else None
            }


class RetryStrategy:
    """Retry strategy with exponential backoff and jitter"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True,
                 backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.backoff_factor = backoff_factor
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if attempt <= 0:
            return 0
        
        # Exponential backoff
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if retry should be attempted"""
        if attempt >= self.max_attempts:
            return False
        
        # Don't retry circuit breaker errors
        if isinstance(exception, CircuitBreakerError):
            return False
        
        # Don't retry certain exception types
        non_retryable = (TypeError, ValueError, KeyError, AttributeError)
        if isinstance(exception, non_retryable):
            return False
        
        return True


class ResilientCaller:
    """Combines circuit breaker with retry strategy and fallback"""
    
    def __init__(self, 
                 circuit_breaker: CircuitBreaker,
                 retry_strategy: RetryStrategy = None,
                 fallback_func: Callable = None):
        self.circuit_breaker = circuit_breaker
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.fallback_func = fallback_func
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker, retry, and fallback"""
        last_exception = None
        
        for attempt in range(1, self.retry_strategy.max_attempts + 1):
            try:
                return self.circuit_breaker.call(func, *args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if not self.retry_strategy.should_retry(attempt, e):
                    break
                
                if attempt < self.retry_strategy.max_attempts:
                    delay = self.retry_strategy.calculate_delay(attempt)
                    logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_strategy.max_attempts})")
                    time.sleep(delay)
        
        # All retries failed, try fallback
        if self.fallback_func:
            try:
                logger.info(f"Executing fallback for circuit '{self.circuit_breaker.config.name}'")
                
                if self.circuit_breaker.config.enable_metrics:
                    circuit_fallbacks.labels(circuit_name=self.circuit_breaker.config.name).inc()
                
                return self.fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise fallback_error
        
        # No fallback available, raise original exception
        raise last_exception
    
    async def call_async(self, coro: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker, retry, and fallback"""
        last_exception = None
        
        for attempt in range(1, self.retry_strategy.max_attempts + 1):
            try:
                return await self.circuit_breaker.call_async(coro, *args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if not self.retry_strategy.should_retry(attempt, e):
                    break
                
                if attempt < self.retry_strategy.max_attempts:
                    delay = self.retry_strategy.calculate_delay(attempt)
                    logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt}/{self.retry_strategy.max_attempts})")
                    await asyncio.sleep(delay)
        
        # All retries failed, try fallback
        if self.fallback_func:
            try:
                logger.info(f"Executing fallback for circuit '{self.circuit_breaker.config.name}'")
                
                if self.circuit_breaker.config.enable_metrics:
                    circuit_fallbacks.labels(circuit_name=self.circuit_breaker.config.name).inc()
                
                if inspect.iscoroutinefunction(self.fallback_func):
                    return await self.fallback_func(*args, **kwargs)
                else:
                    return self.fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise fallback_error
        
        # No fallback available, raise original exception
        raise last_exception


# Decorator implementations
def circuit_breaker(name: str = None,
                   failure_threshold: int = 5,
                   timeout_duration: float = 60.0,
                   fallback: Callable = None):
    """Decorator for circuit breaker pattern"""
    
    def decorator(func):
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        
        config = CircuitBreakerConfig(
            name=circuit_name,
            failure_threshold=failure_threshold,
            timeout_duration=timeout_duration
        )
        
        breaker = CircuitBreaker(config)
        retry_strategy = RetryStrategy()
        caller = ResilientCaller(breaker, retry_strategy, fallback)
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await caller.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return caller.call(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create circuit breaker by name"""
    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig(name=name)
        _circuit_breakers[name] = CircuitBreaker(config)
    
    return _circuit_breakers[name]


def get_all_circuit_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers"""
    return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}


# Example usage and testing
if __name__ == "__main__":
    import requests
    import random
    
    # Example 1: Decorator usage
    @circuit_breaker(name="external_api", failure_threshold=3, timeout_duration=30)
    def call_external_api(url: str) -> dict:
        """Example function that might fail"""
        # Simulate random failures
        if random.random() < 0.3:  # 30% failure rate
            raise requests.exceptions.ConnectionError("Simulated connection error")
        
        return {"status": "success", "data": "some data"}
    
    # Example 2: Manual circuit breaker usage
    def unreliable_service():
        """Simulate an unreliable service"""
        if random.random() < 0.4:  # 40% failure rate
            raise Exception("Service temporarily unavailable")
        return "Service response"
    
    def fallback_service():
        """Fallback when main service fails"""
        return "Fallback response"
    
    # Configure circuit breaker
    config = CircuitBreakerConfig(
        name="unreliable_service",
        failure_threshold=3,
        timeout_duration=10,
        max_retry_attempts=2
    )
    
    breaker = CircuitBreaker(config)
    retry_strategy = RetryStrategy(max_attempts=3, base_delay=0.5)
    caller = ResilientCaller(breaker, retry_strategy, fallback_service)
    
    # Test the circuit breaker
    print("Testing circuit breaker pattern...")
    
    for i in range(20):
        try:
            if i < 10:
                result = call_external_api("https://api.example.com/data")
            else:
                result = caller.call(unreliable_service)
            
            print(f"Call {i+1}: Success - {result}")
            
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
        
        # Print circuit stats every 5 calls
        if (i + 1) % 5 == 0:
            stats = get_all_circuit_stats()
            for name, stat in stats.items():
                print(f"Circuit '{name}': State={stat['state']}, Failures={stat['failed_calls']}/{stat['total_calls']}")
            print("-" * 50)
        
        time.sleep(0.5)
    
    print("\nFinal circuit breaker statistics:")
    for name, stats in get_all_circuit_stats().items():
        print(f"{name}: {stats}")