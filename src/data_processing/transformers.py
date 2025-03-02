import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from datetime import datetime

class DataTransformer:
    """Handles data transformations for different column types"""

    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.encoding_maps = {}  # Store encoding information for inverse transform

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
            transformed = pd.Series(encoder.fit_transform(data), name=data.name)
            self.encoding_maps[data.name] = {
                'method': 'label',
                'categories': list(encoder.classes_),
                'num_categories': len(encoder.classes_)
            }
            return transformed
        elif method == 'onehot':
            # Get dummy variables
            dummies = pd.get_dummies(data, prefix=data.name)
            # Store encoding information for inverse transform
            self.encoding_maps[data.name] = {
                'method': 'onehot',
                'columns': list(dummies.columns),
                'original_categories': list(data.unique()),
                'num_categories': len(data.unique())
            }
            # For one-hot encoding, we return just the first encoded column
            return pd.Series(data.factorize()[0], name=data.name)
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
        encoding_info = self.encoding_maps.get(data.name)
        if encoding_info is None:
            raise ValueError(f"No encoding information found for column {data.name}")

        # Clamp values to valid range and convert to integers
        num_categories = encoding_info['num_categories']
        clamped_values = np.clip(data.values, 0, num_categories - 1).astype(int)

        if encoding_info['method'] == 'label':
            encoder = self.encoders.get(data.name)
            if encoder is None:
                raise ValueError(f"No encoder found for column {data.name}")
            categories = encoding_info['categories']
            return pd.Series(categories[clamped_values], name=data.name)
        elif encoding_info['method'] == 'onehot':
            categories = encoding_info['original_categories']
            return pd.Series(categories[clamped_values], name=data.name)

        raise ValueError(f"Unknown encoding method for column {data.name}")