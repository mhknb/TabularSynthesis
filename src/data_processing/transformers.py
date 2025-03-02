import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from datetime import datetime

class DataTransformer:
    """Handles data transformations for different column types"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        
    def transform_continuous(self, data: pd.Series, method: str = 'minmax') -> pd.Series:
        """Transform continuous data using specified method"""
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        self.scalers[data.name] = scaler
        return pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), name=data.name)
    
    def transform_categorical(self, data: pd.Series, method: str = 'label') -> pd.Series:
        """Transform categorical data using specified method"""
        if method == 'label':
            encoder = LabelEncoder()
            self.encoders[data.name] = encoder
            return pd.Series(encoder.fit_transform(data), name=data.name)
        elif method == 'onehot':
            return pd.get_dummies(data)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def transform_datetime(self, data: pd.Series) -> pd.DataFrame:
        """Transform datetime into multiple features"""
        dt_series = pd.to_datetime(data)
        return pd.DataFrame({
            f"{data.name}_year": dt_series.dt.year,
            f"{data.name}_month": dt_series.dt.month,
            f"{data.name}_day": dt_series.dt.day,
            f"{data.name}_dayofweek": dt_series.dt.dayofweek
        })
    
    def inverse_transform_continuous(self, data: pd.Series) -> pd.Series:
        """Inverse transform continuous data"""
        scaler = self.scalers.get(data.name)
        if scaler is None:
            raise ValueError(f"No scaler found for column {data.name}")
        return pd.Series(scaler.inverse_transform(data.values.reshape(-1, 1)).flatten(), name=data.name)
    
    def inverse_transform_categorical(self, data: pd.Series) -> pd.Series:
        """Inverse transform categorical data"""
        encoder = self.encoders.get(data.name)
        if encoder is None:
            raise ValueError(f"No encoder found for column {data.name}")
        return pd.Series(encoder.inverse_transform(data.astype(int)), name=data.name)
