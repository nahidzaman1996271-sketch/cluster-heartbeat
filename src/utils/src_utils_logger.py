"""
Logging utilities for Cluster Heartbeat.
Configures structured logging with JSON format and context management.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback
import os
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def __init__(self, include_extra: bool = True, **kwargs):
        """
        Initialize JSON formatter.
        
        Args:
            include_extra: Whether to include extra fields
            **kwargs: Additional formatter arguments
        """
        super().__init__(**kwargs)
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'name': record.name,
            'process': record.process,
            'thread': record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_tb(record.exc_info[2]))
            }
        
        # Add extra fields
        if self.include_extra and hasattr(record, 'extra'):
            if isinstance(record.extra, dict):
                log_entry['extra'] = record.extra
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in [
                'args', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'levelname', 'levelno', 'lineno', 'module',
                'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
                'relativeCreated', 'stack_info', 'thread', 'threadName',
                'extra', 'message', 'asctime'
            ]:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def setup_logging(
    level: Union[str, int] = 'INFO',
    log_file: Optional[str] = None,
    format_type: str = 'json',
    max_size: int = 10485760,  # 10 MB
    backup_count: int = 10,
    console_output: bool = True
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to log file
        format_type: 'json' or 'text'
        max_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
    """
    # Convert level string to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if format_type == 'json':
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        level: Optional logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if level:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    
    return logger


class LoggerContext:
    """
    Context manager for adding extra context to logs.
    """
    
    def __init__(self, logger: Union[logging.Logger, str], **kwargs):
        """
        Initialize logger context.
        
        Args:
            logger: Logger instance or name
            **kwargs: Extra fields to add to logs
        """
        if isinstance(logger, str):
            self.logger = get_logger(logger)
        else:
            self.logger = logger
        
        self.extra = kwargs
        self._old_extra = None
    
    def __enter__(self):
        """Enter context."""
        # Store old extra fields
        if hasattr(self.logger, 'extra'):
            self._old_extra = self.logger.extra
        
        # Set new extra fields
        self.logger.extra = {**(self._old_extra or {}), **self.extra}
        
        # Add filter to include extra fields
        self._filter = self._create_filter(self.logger.extra)
        self.logger.addFilter(self._filter)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Remove filter
        self.logger.removeFilter(self._filter)
        
        # Restore old extra fields
        if self._old_extra is not None:
            self.logger.extra = self._old_extra
        elif hasattr(self.logger, 'extra'):
            delattr(self.logger, 'extra')
    
    def _create_filter(self, extra: Dict[str, Any]):
        """Create filter to add extra fields."""
        class ExtraFilter(logging.Filter):
            def filter(self, record):
                for key, value in extra.items():
                    setattr(record, key, value)
                return True
        return ExtraFilter()


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter with extra context.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """
        Initialize logger adapter.
        
        Args:
            logger: Logger instance
            extra: Extra context fields
        """
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message and add extra context.
        
        Args:
            msg: Log message
            kwargs: Additional arguments
            
        Returns:
            Tuple of (msg, kwargs)
        """
        kwargs.setdefault('extra', {})
        kwargs['extra'].update(self.extra)
        return msg, kwargs


# Default logger
logger = get_logger('cluster_heartbeat')


# Convenience functions
def log_exception(logger: logging.Logger, e: Exception, message: str = None) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        e: Exception to log
        message: Optional custom message
    """
    if message:
        logger.error(f"{message}: {str(e)}", exc_info=True)
    else:
        logger.error(f"Exception: {str(e)}", exc_info=True)


def log_startup(logger: logging.Logger, service_name: str, config: Dict[str, Any]) -> None:
    """
    Log service startup information.
    
    Args:
        logger: Logger instance
        service_name: Name of service
        config: Configuration dictionary
    """
    logger.info(f"Starting {service_name}")
    logger.info(f"Environment: {config.get('project', {}).get('environment', 'development')}")
    logger.info(f"Version: {config.get('project', {}).get('version', '1.0.0')}")
    logger.info(f"Debug mode: {config.get('project', {}).get('debug', False)}")
    logger.info(f"Log level: {logging.getLevelName(logger.level)}")