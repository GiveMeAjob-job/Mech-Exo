"""
Structured logging configuration for execution monitoring

Provides JSON-formatted logging with structured fields for:
- Order lifecycle tracking
- Performance monitoring  
- Risk and safety events
- System health monitoring
"""

import logging
import logging.config
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps
import os


@dataclass
class ExecutionContext:
    """Context for execution logging"""
    session_id: str
    trading_mode: str
    account_id: Optional[str] = None
    strategy: Optional[str] = None
    component: Optional[str] = None
    
    @classmethod
    def create(cls, **kwargs) -> 'ExecutionContext':
        """Create new execution context"""
        session_id = kwargs.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
        trading_mode = kwargs.get('trading_mode', os.getenv('EXO_MODE', 'unknown'))
        
        return cls(
            session_id=session_id,
            trading_mode=trading_mode,
            account_id=kwargs.get('account_id'),
            strategy=kwargs.get('strategy'),
            component=kwargs.get('component')
        )


class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, extra_fields: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add thread/process info if relevant
        import threading
        if hasattr(threading, 'main_thread') and record.thread != threading.main_thread().ident:
            log_data['thread_id'] = record.thread
        elif hasattr(record, 'thread') and record.thread:
            log_data['thread_id'] = record.thread
        if hasattr(record, 'process') and record.process:
            log_data['process_id'] = record.process
        
        # Add extra fields from formatter config
        log_data.update(self.extra_fields)
        
        # Add extra fields from log record
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                try:
                    # Only include JSON-serializable values
                    json.dumps(value)
                    extra_data[key] = value
                except (TypeError, ValueError):
                    extra_data[key] = str(value)
        
        if extra_data:
            log_data['extra'] = extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        return json.dumps(log_data, default=str, separators=(',', ':'))


class ExecutionLogger:
    """
    Structured logger for execution monitoring
    
    Provides methods for logging execution events with consistent structure
    """
    
    def __init__(self, logger_name: str, context: Optional[ExecutionContext] = None):
        self.logger = logging.getLogger(logger_name)
        self.context = context or ExecutionContext.create(component=logger_name.split('.')[-1])
    
    def _log_structured(self, level: int, event_type: str, message: str, **kwargs):
        """Log structured event with context"""
        log_data = {
            'event_type': event_type,
            'session_id': self.context.session_id,
            'trading_mode': self.context.trading_mode,
            'component': self.context.component,
            **kwargs
        }
        
        # Add context fields if available
        if self.context.account_id:
            log_data['account_id'] = self.context.account_id
        if self.context.strategy:
            log_data['strategy'] = self.context.strategy
        
        # Add timestamp
        log_data['event_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        self.logger.log(level, message, extra=log_data)
    
    def order_event(self, event_type: str, order_id: str, symbol: str, 
                   message: str, **kwargs):
        """Log order lifecycle event"""
        self._log_structured(
            level=logging.INFO,
            event_type=f"order.{event_type}",
            message=message,
            order_id=order_id,
            symbol=symbol,
            **kwargs
        )
    
    def fill_event(self, fill_id: str, order_id: str, symbol: str, 
                  quantity: int, price: float, message: str, **kwargs):
        """Log fill event"""
        self._log_structured(
            level=logging.INFO,
            event_type="execution.fill",
            message=message,
            fill_id=fill_id,
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            **kwargs
        )
    
    def performance_event(self, metric_name: str, value: float, unit: str,
                         message: str, **kwargs):
        """Log performance metric"""
        self._log_structured(
            level=logging.INFO,
            event_type="performance.metric",
            message=message,
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )
    
    def risk_event(self, risk_type: str, severity: str, message: str, **kwargs):
        """Log risk event"""
        level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity.lower(), logging.WARNING)
        
        self._log_structured(
            level=level,
            event_type=f"risk.{risk_type}",
            message=message,
            risk_severity=severity,
            **kwargs
        )
    
    def safety_event(self, safety_type: str, action: str, message: str, **kwargs):
        """Log safety valve event"""
        self._log_structured(
            level=logging.WARNING,
            event_type=f"safety.{safety_type}",
            message=message,
            safety_action=action,
            **kwargs
        )
    
    def system_event(self, system: str, status: str, message: str, **kwargs):
        """Log system status event"""
        level = logging.ERROR if status in ['error', 'failed', 'disconnected'] else logging.INFO
        
        self._log_structured(
            level=level,
            event_type=f"system.{system}",
            message=message,
            system_status=status,
            **kwargs
        )
    
    def error_event(self, error_type: str, error_message: str, message: str, **kwargs):
        """Log error event"""
        self._log_structured(
            level=logging.ERROR,
            event_type=f"error.{error_type}",
            message=message,
            error_message=error_message,
            **kwargs
        )


@contextmanager
def execution_timer(logger: ExecutionLogger, operation: str, **context):
    """Context manager to time execution operations"""
    start_time = time.perf_counter()
    start_timestamp = datetime.now(timezone.utc)
    
    logger._log_structured(
        level=logging.DEBUG,
        event_type="timing.start",
        message=f"Starting {operation}",
        operation=operation,
        start_timestamp=start_timestamp.isoformat(),
        **context
    )
    
    try:
        yield
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        logger.performance_event(
            metric_name=f"{operation}_duration",
            value=duration_ms,
            unit="milliseconds",
            message=f"Completed {operation}",
            operation=operation,
            success=success,
            error_message=error_msg,
            **context
        )


def timed_execution(operation_name: str = None):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to get logger from self if method
            logger = None
            if args and hasattr(args[0], 'execution_logger'):
                logger = args[0].execution_logger
            elif args and hasattr(args[0], 'logger'):
                # Fallback to regular logger
                logger_name = f"{args[0].__class__.__module__}.{args[0].__class__.__name__}"
                logger = ExecutionLogger(logger_name)
            else:
                logger = ExecutionLogger("execution.timing")
            
            op_name = operation_name or f"{func.__name__}"
            
            with execution_timer(logger, op_name, function=func.__name__):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Try to get logger from self if method
            logger = None
            if args and hasattr(args[0], 'execution_logger'):
                logger = args[0].execution_logger
            elif args and hasattr(args[0], 'logger'):
                # Fallback to regular logger
                logger_name = f"{args[0].__class__.__module__}.{args[0].__class__.__name__}"
                logger = ExecutionLogger(logger_name)
            else:
                logger = ExecutionLogger("execution.timing")
            
            op_name = operation_name or f"{func.__name__}"
            
            with execution_timer(logger, op_name, function=func.__name__):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'async' in func.__code__.co_name:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def configure_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    json_format: bool = True,
    extra_fields: Optional[Dict[str, Any]] = None
):
    """
    Configure structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console_output: Whether to output to console
        json_format: Whether to use JSON formatting
        extra_fields: Extra fields to include in all log messages
    """
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {},
        'handlers': {},
        'loggers': {
            'mech_exo': {
                'level': log_level,
                'handlers': [],
                'propagate': False
            },
            'root': {
                'level': log_level,
                'handlers': []
            }
        }
    }
    
    # Configure formatters
    if json_format:
        config['formatters']['structured'] = {
            '()': StructuredLogFormatter,
            'extra_fields': extra_fields or {}
        }
        formatter_name = 'structured'
    else:
        config['formatters']['standard'] = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        formatter_name = 'standard'
    
    # Configure console handler
    if console_output:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': formatter_name,
            'stream': 'ext://sys.stdout'
        }
        config['loggers']['mech_exo']['handlers'].append('console')
        config['loggers']['root']['handlers'].append('console')
    
    # Configure file handler
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': formatter_name,
            'filename': log_file,
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5
        }
        config['loggers']['mech_exo']['handlers'].append('file')
        config['loggers']['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log configuration success
    logger = ExecutionLogger("mech_exo.logging")
    logger.system_event(
        system="logging",
        status="configured",
        message="Structured logging configured",
        log_level=log_level,
        json_format=json_format,
        log_file=log_file,
        console_output=console_output
    )


def get_execution_logger(name: str, context: Optional[ExecutionContext] = None) -> ExecutionLogger:
    """Get an execution logger with optional context"""
    return ExecutionLogger(name, context)


# Global context for session tracking
_current_context: Optional[ExecutionContext] = None


def set_execution_context(context: ExecutionContext):
    """Set global execution context"""
    global _current_context
    _current_context = context


def get_execution_context() -> Optional[ExecutionContext]:
    """Get current execution context"""
    return _current_context


def create_session_context(**kwargs) -> ExecutionContext:
    """Create and set new session context"""
    context = ExecutionContext.create(**kwargs)
    set_execution_context(context)
    return context