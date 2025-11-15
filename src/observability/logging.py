"""
JSON-structured logging infrastructure for multi-agent MCTS framework.

Provides:
- JSON-structured logging via logging.config.dictConfig
- Per-module loggers with proper hierarchy
- Correlation IDs for request tracking
- Log levels configurable via environment/settings
- Performance metrics in logs (timing, memory)
- Safe sanitization (no secrets in logs)
"""

import json
import logging
import logging.config
import os
import re
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Optional

import psutil

# Context variable for correlation ID tracking across async calls
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_metadata: ContextVar[dict] = ContextVar("request_metadata", default={})


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = _correlation_id.get()
    if cid is None:
        cid = str(uuid.uuid4())
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    _correlation_id.set(cid)


def set_request_metadata(metadata: dict) -> None:
    """Set request metadata for current context."""
    _request_metadata.set(metadata)


def get_request_metadata() -> dict:
    """Get request metadata for current context."""
    return _request_metadata.get()


# Patterns for sensitive data sanitization
SENSITIVE_PATTERNS = [
    (re.compile(r'("?api[_-]?key"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?password"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?secret"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?token"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?authorization"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'("?aws[_-]?secret[_-]?access[_-]?key"?\s*[:=]\s*)"[^"]*"', re.IGNORECASE), r'\1"***REDACTED***"'),
    (re.compile(r'(Bearer\s+)\S+', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'(Basic\s+)\S+', re.IGNORECASE), r'\1***REDACTED***'),
]


def sanitize_message(message: str) -> str:
    """Sanitize sensitive data from log messages."""
    for pattern, replacement in SENSITIVE_PATTERNS:
        message = pattern.sub(replacement, message)
    return message


def sanitize_dict(data: dict) -> dict:
    """Recursively sanitize sensitive data from dictionaries."""
    sensitive_keys = {
        "api_key", "apikey", "password", "secret", "token", "authorization",
        "auth", "credentials", "aws_secret_access_key", "private_key",
    }

    result = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in sensitive_keys:
            result[key] = "***REDACTED***"
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str):
            result[key] = sanitize_message(value)
        else:
            result[key] = value
    return result


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID and request metadata to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        record.request_metadata = get_request_metadata()
        return True


class PerformanceMetricsFilter(logging.Filter):
    """Add performance metrics to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        process = psutil.Process()
        record.memory_mb = process.memory_info().rss / (1024 * 1024)
        record.cpu_percent = process.cpu_percent()
        record.thread_count = process.num_threads()
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON with comprehensive metadata."""

    def __init__(self, include_hostname: bool = True, include_process: bool = True):
        super().__init__()
        self.include_hostname = include_hostname
        self.include_process = include_process
        if include_hostname:
            import socket
            self.hostname = socket.gethostname()
        else:
            self.hostname = None

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_message(record.getMessage()),
            "correlation_id": getattr(record, "correlation_id", None),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add hostname if configured
        if self.hostname:
            log_data["hostname"] = self.hostname

        # Add process info if configured
        if self.include_process:
            log_data["process"] = {
                "id": record.process,
                "name": record.processName,
                "thread_id": record.thread,
                "thread_name": record.threadName,
            }

        # Add performance metrics if available
        if hasattr(record, "memory_mb"):
            log_data["performance"] = {
                "memory_mb": round(getattr(record, "memory_mb", 0), 2),
                "cpu_percent": round(getattr(record, "cpu_percent", 0), 2),
                "thread_count": getattr(record, "thread_count", 0),
            }

        # Add request metadata if available
        request_metadata = getattr(record, "request_metadata", {})
        if request_metadata:
            log_data["request"] = sanitize_dict(request_metadata)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add any extra fields (sanitized)
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs", "message",
                "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info",
                "correlation_id", "request_metadata", "memory_mb", "cpu_percent",
                "thread_count", "taskName",
            }:
                if isinstance(value, dict):
                    extra_fields[key] = sanitize_dict(value)
                elif isinstance(value, str):
                    extra_fields[key] = sanitize_message(value)
                else:
                    extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    include_performance_metrics: bool = True,
    json_output: bool = True,
    include_hostname: bool = True,
    include_process: bool = True,
) -> None:
    """
    Configure JSON-structured logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  Defaults to LOG_LEVEL env var or INFO.
        log_file: Optional file path for log output.
        include_performance_metrics: Include memory/CPU metrics in logs.
        json_output: Use JSON formatter (default True).
        include_hostname: Include hostname in logs.
        include_process: Include process/thread info in logs.
    """
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation_id": {
                "()": CorrelationIdFilter,
            },
        },
        "formatters": {
            "json": {
                "()": JSONFormatter,
                "include_hostname": include_hostname,
                "include_process": include_process,
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(correlation_id)s): %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json" if json_output else "standard",
                "filters": ["correlation_id"],
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            # Root logger
            "": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            # Framework loggers with hierarchy
            "mcts": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "mcts.framework": {
                "level": log_level,
                "propagate": True,
            },
            "mcts.agents": {
                "level": log_level,
                "propagate": True,
            },
            "mcts.observability": {
                "level": log_level,
                "propagate": True,
            },
            "mcts.storage": {
                "level": log_level,
                "propagate": True,
            },
            # Third-party loggers (quieter)
            "httpx": {
                "level": "WARNING",
                "propagate": True,
            },
            "opentelemetry": {
                "level": "WARNING",
                "propagate": True,
            },
            "aioboto3": {
                "level": "WARNING",
                "propagate": True,
            },
            "botocore": {
                "level": "WARNING",
                "propagate": True,
            },
        },
    }

    # Add performance metrics filter if requested
    if include_performance_metrics:
        config["filters"]["performance_metrics"] = {
            "()": PerformanceMetricsFilter,
        }
        config["handlers"]["console"]["filters"].append("performance_metrics")

    # Add file handler if requested
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json" if json_output else "standard",
            "filters": ["correlation_id"],
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        if include_performance_metrics:
            config["handlers"]["file"]["filters"].append("performance_metrics")

        # Add file handler to all loggers
        config["loggers"][""]["handlers"].append("file")
        config["loggers"]["mcts"]["handlers"].append("file")

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Uses hierarchical naming under 'mcts' root logger.
    Example: get_logger("framework.graph") returns logger "mcts.framework.graph"

    Args:
        name: Logger name (will be prefixed with 'mcts.')

    Returns:
        Configured logger instance
    """
    if not name.startswith("mcts."):
        name = f"mcts.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding temporary log context."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._old_metadata = None

    def __enter__(self):
        self._old_metadata = get_request_metadata().copy()
        new_metadata = {**self._old_metadata, **self.kwargs}
        set_request_metadata(new_metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_request_metadata(self._old_metadata)
        return False


def log_execution_time(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (defaults to function's module logger)
        level: Log level for timing message

    Example:
        @log_execution_time()
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                memory_delta_mb = (psutil.Process().memory_info().rss - start_memory) / (1024 * 1024)

                logger.log(
                    level,
                    f"Function {func.__name__} completed",
                    extra={
                        "timing": {
                            "function": func.__name__,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "success": success,
                            "error": error,
                            "memory_delta_mb": round(memory_delta_mb, 2),
                        }
                    }
                )

            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.perf_counter()
            start_memory = psutil.Process().memory_info().rss

            try:
                result = await func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                memory_delta_mb = (psutil.Process().memory_info().rss - start_memory) / (1024 * 1024)

                logger.log(
                    level,
                    f"Async function {func.__name__} completed",
                    extra={
                        "timing": {
                            "function": func.__name__,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "success": success,
                            "error": error,
                            "memory_delta_mb": round(memory_delta_mb, 2),
                        }
                    }
                )

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Import asyncio for decorator
import asyncio
