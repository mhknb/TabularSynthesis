
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats

class OutlierDetector:
    """Detects and filters outliers in tabular data"""
    
    @staticmethod
    def detect_statistical_outliers(df, numerical_cols, z_threshold=3.0):
        """
        Detect outliers using Z-score method
        
        Args:
            df: DataFrame containing the data
            numerical_cols: List of numerical column names
            z_threshold: Z-score threshold (default: 3.0)
            
        Returns:
            DataFrame with outlier flags for each row
        """
        outlier_flags = pd.DataFrame(index=df.index)
        outlier_flags['is_outlier'] = False
        
        for col in numerical_cols:
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
            outlier_flags[f'{col}_outlier'] = z_scores > z_threshold
            outlier_flags['is_outlier'] |= outlier_flags[f'{col}_outlier']
            
        return outlier_flags
    
    @staticmethod
    def detect_isolation_forest_outliers(df, numerical_cols, contamination=0.05):
        """
        Detect outliers using Isolation Forest algorithm
        
        Args:
            df: DataFrame containing the data
            numerical_cols: List of numerical column names
            contamination: Expected proportion of outliers (default: 0.05)
            
        Returns:
            DataFrame with outlier flags for each row
        """
        if not numerical_cols:
            return pd.DataFrame(index=df.index, data={'is_outlier': False})
            
        # Select numerical features
        X = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_predictions = iso_forest.fit_predict(X)
        
        # Create outlier flags (-1 for outliers, 1 for inliers)
        outlier_flags = pd.DataFrame(index=df.index)
        outlier_flags['is_outlier'] = outlier_predictions == -1
        
        return outlier_flags
    
    @staticmethod
    def detect_iqr_outliers(df, numerical_cols, k=1.5):
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            df: DataFrame containing the data
            numerical_cols: List of numerical column names
            k: IQR multiplier (default: 1.5)
            
        Returns:
            DataFrame with outlier flags for each row
        """
        outlier_flags = pd.DataFrame(index=df.index)
        outlier_flags['is_outlier'] = False
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            outlier_flags[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_flags['is_outlier'] |= outlier_flags[f'{col}_outlier']
        
        return outlier_flags
    
    @staticmethod
    def remove_outliers(df, method='iqr', **kwargs):
        """
        Remove outliers from DataFrame
        
        Args:
            df: DataFrame containing the data
            method: Outlier detection method ('iqr', 'zscore', or 'isolation_forest')
            **kwargs: Additional parameters for the outlier detection method
            
        Returns:
            DataFrame with outliers removed
        """
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if method == 'iqr':
            outlier_flags = OutlierDetector.detect_iqr_outliers(
                df, numerical_cols, k=kwargs.get('k', 1.5)
            )
        elif method == 'zscore':
            outlier_flags = OutlierDetector.detect_statistical_outliers(
                df, numerical_cols, z_threshold=kwargs.get('z_threshold', 3.0)
            )
        elif method == 'isolation_forest':
            outlier_flags = OutlierDetector.detect_isolation_forest_outliers(
                df, numerical_cols, contamination=kwargs.get('contamination', 0.05)
            )
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Get clean data (non-outliers)
        clean_df = df[~outlier_flags['is_outlier']].copy()
        outlier_count = outlier_flags['is_outlier'].sum()
        outlier_percentage = (outlier_count / len(df)) * 100
        
        print(f"Removed {outlier_count} outliers ({outlier_percentage:.2f}% of data)")
        
        return clean_df
