"""Advanced data transformation module for synthetic data generation"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.impute import SimpleImputer

class AdvancedTransformer:
    """Advanced data transformation class with multiple scaling and encoding options"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.categorical_columns = []
        self.numerical_columns = []
        
    def fit_transform(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """
        Fit and transform data using specified transformations
        
        Args:
            data: Input DataFrame
            config: Configuration dictionary containing:
                - scaling_method: str ('minmax', 'standard', 'robust', 'power', 'quantile')
                - encoding_method: str ('label', 'onehot', 'target')
                - handle_outliers: bool
                - handle_missing: bool
                - feature_engineering: list of feature engineering operations
        """
        transformed_data = data.copy()
        
        # Identify column types
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
        
        # Handle missing values if configured
        if config.get('handle_missing', False):
            transformed_data = self._handle_missing_values(transformed_data)
            
        # Handle outliers if configured
        if config.get('handle_outliers', False):
            transformed_data = self._handle_outliers(transformed_data)
            
        # Apply scaling to numerical columns
        scaling_method = config.get('scaling_method', 'standard')
        transformed_data = self._apply_scaling(transformed_data, scaling_method)
        
        # Apply encoding to categorical columns
        encoding_method = config.get('encoding_method', 'onehot')
        transformed_data = self._apply_encoding(transformed_data, encoding_method)
        
        # Apply feature engineering if configured
        if 'feature_engineering' in config:
            transformed_data = self._apply_feature_engineering(transformed_data, config['feature_engineering'])
            
        return transformed_data
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data to original scale"""
        restored_data = data.copy()
        
        # Inverse transform numerical columns
        for col, scaler in self.scalers.items():
            if col in restored_data.columns:
                restored_data[col] = scaler.inverse_transform(restored_data[[col]])
                
        # TODO: Add inverse transform for encodings
        
        return restored_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies"""
        for col in self.numerical_columns:
            if data[col].isnull().any():
                self.imputers[col] = SimpleImputer(strategy='mean')
                data[col] = self.imputers[col].fit_transform(data[[col]])
                
        for col in self.categorical_columns:
            if data[col].isnull().any():
                self.imputers[col] = SimpleImputer(strategy='most_frequent')
                data[col] = self.imputers[col].fit_transform(data[[col]])
                
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        for col in self.numerical_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower_bound, upper_bound)
        return data
    
    def _apply_scaling(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply specified scaling method to numerical columns"""
        scaler_map = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'power': PowerTransformer(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        
        scaler = scaler_map.get(method, StandardScaler())
        
        for col in self.numerical_columns:
            self.scalers[col] = scaler.__class__()
            data[col] = self.scalers[col].fit_transform(data[[col]])
            
        return data
    
    def _apply_encoding(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply specified encoding method to categorical columns"""
        if method == 'label':
            for col in self.categorical_columns:
                data[col] = pd.factorize(data[col])[0]
        elif method == 'onehot':
            data = pd.get_dummies(data, columns=self.categorical_columns)
            
        return data
    
    def _apply_feature_engineering(self, data: pd.DataFrame, operations: list) -> pd.DataFrame:
        """Apply specified feature engineering operations"""
        for operation in operations:
            if operation == 'polynomial':
                for col in self.numerical_columns:
                    data[f"{col}_squared"] = data[col] ** 2
            elif operation == 'log':
                for col in self.numerical_columns:
                    if (data[col] > 0).all():
                        data[f"{col}_log"] = np.log(data[col])
                        
        return data
