"""
Data validation utilities for Cluster Heartbeat.
Validates cluster metrics, node data, and job data.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """
    Result of data validation.
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)
    
    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion."""
        self.suggestions.append(suggestion)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'metadata': self.metadata
        }


class DataValidator:
    """
    Validates cluster metrics data.
    """
    
    def __init__(self):
        """Initialize data validator."""
        self.required_fields = [
            'gpu_utilization',
            'memory_utilization',
            'gpu_temperature'
        ]
        
        self.optional_fields = [
            'power_consumption',
            'ecc_errors',
            'xid_errors',
            'cpu_usage',
            'ram_usage',
            'network_throughput',
            'disk_io',
            'job_runtime',
            'queue_length',
            'active_processes',
            'node_id',
            'job_id',
            'timestamp'
        ]
        
        # Field validation ranges
        self.field_ranges = {
            'gpu_utilization': (0.0, 1.0),
            'memory_utilization': (0.0, 1.0),
            'gpu_temperature': (-50.0, 150.0),
            'power_consumption': (0.0, 1000.0),
            'ecc_errors': (0.0, float('inf')),
            'xid_errors': (0.0, float('inf')),
            'cpu_usage': (0.0, 1.0),
            'ram_usage': (0.0, 1.0),
            'network_throughput': (0.0, float('inf')),
            'disk_io': (0.0, float('inf')),
            'job_runtime': (0.0, float('inf')),
            'queue_length': (0.0, float('inf')),
            'active_processes': (0.0, float('inf'))
        }
        
        self.field_types = {
            'gpu_utilization': float,
            'memory_utilization': float,
            'gpu_temperature': float,
            'power_consumption': float,
            'ecc_errors': int,
            'xid_errors': int,
            'cpu_usage': float,
            'ram_usage': float,
            'network_throughput': float,
            'disk_io': float,
            'job_runtime': float,
            'queue_length': float,
            'active_processes': int,
            'node_id': str,
            'job_id': str,
            'timestamp': float
        }
        
        logger.info("DataValidator initialized")
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult()
        
        if df.empty:
            result.add_error("DataFrame is empty")
            return result
        
        # Check required columns
        missing_columns = [col for col in self.required_fields if col not in df.columns]
        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")
        
        # Check each row
        for idx, row in df.iterrows():
            row_result = self.validate_row(row.to_dict())
            if not row_result.is_valid:
                result.add_error(f"Row {idx}: {', '.join(row_result.errors)}")
            if row_result.warnings:
                for warning in row_result.warnings:
                    result.add_warning(f"Row {idx}: {warning}")
        
        return result
    
    def validate_row(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single row of data.
        
        Args:
            data: Data dictionary
            
        Returns:
            Validation result
        """
        result = ValidationResult()
        
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
            elif data[field] is None:
                result.add_error(f"Field {field} has null value")
            else:
                # Validate field type
                if field in self.field_types:
                    expected_type = self.field_types[field]
                    if not isinstance(data[field], expected_type):
                        result.add_warning(
                            f"Field {field} expected {expected_type.__name__}, "
                            f"got {type(data[field]).__name__}"
                        )
                
                # Validate field range
                if field in self.field_ranges:
                    min_val, max_val = self.field_ranges[field]
                    if data[field] < min_val or data[field] > max_val:
                        result.add_warning(
                            f"Field {field} value {data[field]} outside range "
                            f"[{min_val}, {max_val}]"
                        )
        
        # Check optional fields
        for field in self.optional_fields:
            if field in data and data[field] is not None:
                if field in self.field_types:
                    expected_type = self.field_types[field]
                    if not isinstance(data[field], expected_type):
                        result.add_warning(
                            f"Optional field {field} expected {expected_type.__name__}, "
                            f"got {type(data[field]).__name__}"
                        )
        
        # Validate timestamp
        if 'timestamp' in data and data['timestamp'] is not None:
            timestamp = data['timestamp']
            if isinstance(timestamp, (int, float)):
                # Check if timestamp is reasonable (not in future by more than 1 hour)
                now = time.time()
                if timestamp > now + 3600:
                    result.add_warning(f"Timestamp is more than 1 hour in the future")
                elif timestamp < now - 86400 * 365:  # 1 year
                    result.add_warning(f"Timestamp is more than 1 year in the past")
        
        return result
    
    def validate_list(self, data_list: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate a list of data dictionaries.
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            Validation result
        """
        result = ValidationResult()
        
        if not data_list:
            result.add_error("Data list is empty")
            return result
        
        for i, data in enumerate(data_list):
            row_result = self.validate_row(data)
            if not row_result.is_valid:
                result.add_error(f"Item {i}: {', '.join(row_result.errors)}")
            if row_result.warnings:
                for warning in row_result.warnings:
                    result.add_warning(f"Item {i}: {warning}")
        
        return result
    
    def validate_metric_values(self, metrics: Dict[str, float]) -> ValidationResult:
        """
        Validate metric values.
        
        Args:
            metrics: Metric values dictionary
            
        Returns:
            Validation result
        """
        result = ValidationResult()
        
        for field, value in metrics.items():
            # Check range
            if field in self.field_ranges:
                min_val, max_val = self.field_ranges[field]
                if value < min_val or value > max_val:
                    result.add_warning(
                        f"Metric {field} value {value} outside range [{min_val}, {max_val}]"
                    )
            
            # Check for NaN or Inf
            if np.isnan(value):
                result.add_error(f"Metric {field} is NaN")
            elif np.isinf(value):
                result.add_error(f"Metric {field} is infinite")
        
        return result
    
    def check_data_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data completeness.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Completeness statistics
        """
        total_rows = len(df)
        completeness = {}
        
        for col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = {
                'non_null_count': int(non_null),
                'non_null_percentage': float(non_null / total_rows * 100) if total_rows > 0 else 0,
                'null_count': int(total_rows - non_null),
                'null_percentage': float((total_rows - non_null) / total_rows * 100) if total_rows > 0 else 0
            }
        
        return {
            'total_rows': total_rows,
            'columns': completeness,
            'overall_completeness': float(df.notna().sum().sum() / (total_rows * len(df.columns)) * 100) if total_rows > 0 else 0
        }
    
    def suggest_fixes(self, data: Dict[str, Any]) -> List[str]:
        """
        Suggest fixes for invalid data.
        
        Args:
            data: Data dictionary
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        for field, value in data.items():
            if field in self.field_ranges:
                min_val, max_val = self.field_ranges[field]
                if isinstance(value, (int, float)):
                    if value < min_val:
                        suggestions.append(
                            f"Field {field} value {value} is below minimum {min_val}. "
                            f"Consider using {min_val} as minimum."
                        )
                    elif value > max_val:
                        suggestions.append(
                            f"Field {field} value {value} is above maximum {max_val}. "
                            f"Consider using {max_val} as maximum."
                        )
        
        # Check for missing required fields
        for field in self.required_fields:
            if field not in data:
                suggestions.append(f"Add required field: {field}")
            elif data[field] is None:
                suggestions.append(f"Populate required field: {field}")
        
        return suggestions


# Convenience functions
def validate_cluster_metrics(data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> ValidationResult:
    """
    Validate cluster metrics data.
    
    Args:
        data: Data to validate (DataFrame, dict, or list of dicts)
        
    Returns:
        Validation result
    """
    validator = DataValidator()
    
    if isinstance(data, pd.DataFrame):
        return validator.validate_dataframe(data)
    elif isinstance(data, dict):
        return validator.validate_row(data)
    elif isinstance(data, list):
        return validator.validate_list(data)
    else:
        result = ValidationResult()
        result.add_error(f"Unsupported data type: {type(data)}")
        return result


def validate_node_data(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate node data.
    
    Args:
        data: Node data dictionary
        
    Returns:
        Validation result
    """
    validator = DataValidator()
    result = validator.validate_row(data)
    
    # Additional node-specific validation
    if 'node_id' in data and data['node_id']:
        if not isinstance(data['node_id'], (str, int)):
            result.add_warning("node_id should be string or integer")
    
    return result


def validate_job_data(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate job data.
    
    Args:
        data: Job data dictionary
        
    Returns:
        Validation result
    """
    validator = DataValidator()
    result = validator.validate_row(data)
    
    # Additional job-specific validation
    if 'job_id' in data and data['job_id']:
        if not isinstance(data['job_id'], (str, int)):
            result.add_warning("job_id should be string or integer")
    
    if 'job_runtime' in data and data['job_runtime']:
        if data['job_runtime'] < 0:
            result.add_warning("job_runtime should be non-negative")
    
    return result


# Import time for timestamp validation
import time