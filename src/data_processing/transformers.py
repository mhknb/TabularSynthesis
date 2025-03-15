"""
Data transformation utilities for preprocessing different data types
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

class DataTransformer:
    """Handles data transformations for different column types"""

    def __init__(self):
        """Initialize transformation components"""
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.encoding_maps = {}
        self.data_ranges = {}
        self.missing_flags = {}

    def transform_continuous(self, data: pd.Series, method: str = 'minmax') -> pd.Series:
        """Transform continuous data using specified method with missing value handling"""
        if data is None or data.empty:
            raise ValueError(f"Invalid data provided for transformation")

        # Handle missing values
        has_missing = data.isnull().any()
        if has_missing:
            self.missing_flags[data.name] = data.isnull().astype(int)
            imputer = SimpleImputer(strategy='mean')
            data_reshaped = data.values.reshape(-1, 1)
            imputed_data = imputer.fit_transform(data_reshaped).flatten()
            self.imputers[data.name] = imputer
            data_for_transform = pd.Series(imputed_data, name=data.name)
        else:
            data_for_transform = data

        # Store original range
        self.data_ranges[data.name] = {
            'min': data_for_transform.min(),
            'max': data_for_transform.max(),
            'dtype': data.dtype,
            'has_missing': has_missing
        }

        # Always use MinMaxScaler for GAN compatibility
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scalers[data.name] = scaler
        transformed = scaler.fit_transform(data_for_transform.values.reshape(-1, 1)).flatten()

        # Ensure values are strictly between 0 and 1
        transformed = np.clip(transformed, 0, 1)
        return pd.Series(transformed, name=data.name)

    def transform_categorical(self, data: pd.Series, method: str = 'label') -> pd.Series:
        """Transform categorical data with specified method"""
        if data is None or data.empty:
            raise ValueError("Invalid data provided for transformation")

        # Handle missing values
        has_missing = data.isnull().any()
        if has_missing:
            self.missing_flags[data.name] = data.isnull().astype(int)
            imputer = SimpleImputer(strategy='most_frequent')
            data_reshaped = data.values.reshape(-1, 1)
            imputed_data = imputer.fit_transform(data_reshaped).flatten()
            self.imputers[data.name] = imputer
            data_for_transform = pd.Series(imputed_data, name=data.name)
        else:
            data_for_transform = data

        if method == 'label':
            encoder = LabelEncoder()
            self.encoders[data.name] = encoder
            transformed = encoder.fit_transform(data_for_transform.astype(str))

            # Normalize to [0,1] range for GAN
            transformed = transformed / (len(encoder.classes_) - 1)

            self.encoding_maps[data.name] = {
                'method': 'label',
                'categories': encoder.classes_,
                'num_classes': len(encoder.classes_),
                'has_missing': has_missing
            }
            return pd.Series(transformed, name=data.name)

        elif method == 'binary':
            unique_values = data_for_transform.unique()
            n_values = len(unique_values)
            # Normalize binary values to 0,1
            value_map = {val: i/max(1, n_values-1) for i, val in enumerate(unique_values)}

            self.encoding_maps[data.name] = {
                'method': 'binary',
                'mapping': value_map,
                'unique_values': unique_values.tolist(),
                'has_missing': has_missing
            }

            transformed = data_for_transform.map(value_map)
            return pd.Series(transformed, name=data.name)

        elif method == 'onehot':
            dummies = pd.get_dummies(data_for_transform, prefix=data.name)
            self.encoding_maps[data.name] = {
                'method': 'onehot',
                'columns': dummies.columns.tolist(),
                'has_missing': has_missing
            }
            return dummies

        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def transform_datetime(self, data: pd.Series) -> pd.DataFrame:
        """Transform datetime into multiple features"""
        if data is None or data.empty:
            raise ValueError("Invalid data provided for transformation")

        dt_series = pd.to_datetime(data)

        # Extract and normalize features
        year = (dt_series.dt.year - dt_series.dt.year.min()) / max(1, dt_series.dt.year.max() - dt_series.dt.year.min())
        month = (dt_series.dt.month - 1) / 11  # 1-12 -> 0-1
        day = (dt_series.dt.day - 1) / 30  # 1-31 -> 0-1
        dayofweek = dt_series.dt.dayofweek / 6  # 0-6 -> 0-1

        return pd.DataFrame({
            f"{data.name}_year": year,
            f"{data.name}_month": month,
            f"{data.name}_day": day,
            f"{data.name}_dayofweek": dayofweek
        })

    def inverse_transform_continuous(self, data: pd.Series) -> pd.Series:
        """Inverse transform continuous data"""
        scaler = self.scalers.get(data.name)
        if scaler is None:
            return data

        # Handle missing values
        if data.isnull().any():
            data = data.fillna(0)

        # Ensure values are in range [0,1] before inverse transform
        data_clipped = np.clip(data.values, 0, 1)
        transformed = scaler.inverse_transform(data_clipped.reshape(-1, 1)).flatten()

        # Restore original range if available
        data_range = self.data_ranges.get(data.name)
        if data_range:
            if data_range.get('dtype') in [np.int32, np.int64, int]:
                transformed = np.round(transformed).astype(data_range['dtype'])

        return pd.Series(transformed, name=data.name)

    def inverse_transform_categorical(self, data: pd.Series) -> pd.Series:
        """Inverse transform categorical data"""
        if data.isnull().all():
            return data

        encoding_info = self.encoding_maps.get(data.name)
        if encoding_info is None:
            return data

        # Handle missing values
        data_filled = data.fillna(0)

        if encoding_info['method'] == 'label':
            encoder = self.encoders.get(data.name)
            if encoder is None:
                return data

            # Scale back from [0,1] to class indices
            num_classes = encoding_info['num_classes']
            indices = np.round(data_filled * (num_classes - 1)).astype(int)
            indices = np.clip(indices, 0, num_classes - 1)

            return pd.Series(
                encoder.inverse_transform(indices),
                name=data.name
            )

        elif encoding_info['method'] == 'binary':
            # Find closest matching value
            unique_values = encoding_info['unique_values']
            mapping = encoding_info['mapping']
            reverse_map = {v: k for k, v in mapping.items()}

            def find_nearest(val):
                return reverse_map[min(mapping.values(), key=lambda x: abs(x - val))]

            return data_filled.map(find_nearest)

        elif encoding_info['method'] == 'onehot':
            columns = encoding_info.get('columns', [])
            if not columns:
                return data

            max_idx = np.round(data_filled * (len(columns) - 1)).astype(int)
            max_idx = np.clip(max_idx, 0, len(columns) - 1)
            return pd.Series(
                [columns[i].split('_')[-1] for i in max_idx],
                name=data.name
            )

        return data