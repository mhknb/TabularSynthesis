import pandas as pd
import numpy as np
from typing import Tuple, List

def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate input data for common issues"""
    issues = []
    
    # Check for null values
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        issues.append(f"Missing values found in columns: {', '.join(null_cols)}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append(f"Constant columns found: {', '.join(constant_cols)}")
    
    # Check for high cardinality in categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) > 0.5:
            issues.append(f"High cardinality in categorical column: {col}")
    
    return len(issues) == 0, issues

def check_column_types(column_types: dict, df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate column type assignments"""
    issues = []
    
    for col, col_type in column_types.items():
        if col not in df.columns:
            issues.append(f"Column {col} not found in dataset")
            continue
            
        if col_type == 'Continuous':
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} marked as Continuous but contains non-numeric data")
                
        elif col_type == 'Datetime':
            try:
                pd.to_datetime(df[col])
            except:
                issues.append(f"Column {col} marked as Datetime but cannot be converted to datetime")
    
    return len(issues) == 0, issues
