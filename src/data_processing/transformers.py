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

    def transform_continuous(self, data: pd.Series, method: str = 'standard') -> pd.Series:
        """Transform continuous data with specified method"""
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

        # Scale data
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scalers[data.name] = scaler
        transformed = scaler.fit_transform(data_for_transform.values.reshape(-1, 1)).flatten()
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

        if method == 'binary':
            # Get unique values and create binary mapping
            unique_values = data_for_transform.unique()
            n_values = len(unique_values)
            n_bits = int(np.ceil(np.log2(max(2, n_values))))  # At least 1 bit

            # Create binary encodings
            binary_codes = {}
            for i, val in enumerate(unique_values):
                binary_code = format(i, f'0{n_bits}b')
                binary_codes[val] = binary_code

            # Store encoding information for inverse transform
            self.encoding_maps[data.name] = {
                'method': 'binary',
                'mapping': binary_codes,
                'n_bits': n_bits,
                'unique_values': unique_values.tolist(),
                'has_missing': has_missing
            }

            # Convert values to their binary representation
            transformed = data_for_transform.map(lambda x: int(''.join(binary_codes[x]), 2))
            return transformed.astype(np.float64)

        elif method == 'label':
            encoder = LabelEncoder()
            self.encoders[data.name] = encoder
            transformed = pd.Series(
                encoder.fit_transform(data_for_transform.astype(str)),
                name=data.name
            )
            self.encoding_maps[data.name] = {
                'method': 'label',
                'categories': encoder.classes_,
                'has_missing': has_missing
            }
            return transformed.astype(np.float64)

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
            return data

        # Handle missing values
        if data.isnull().any():
            data = data.fillna(0)

        # Inverse transform
        transformed = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()

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

        if encoding_info['method'] == 'binary':
            n_bits = encoding_info['n_bits']
            unique_values = encoding_info['unique_values']

            # Convert integers back to binary strings
            binary_values = data_filled.astype(int).map(
                lambda x: format(x, f'0{n_bits}b')
            )

            # Create reverse mapping from binary string to original value
            reverse_mapping = {
                int(b, 2): val for val, b in encoding_info['mapping'].items()
            }

            # Map back to original values
            return pd.Series(
                binary_values.map(lambda x: unique_values[int(x, 2)]),
                name=data.name
            )

        elif encoding_info['method'] == 'label':
            encoder = self.encoders.get(data.name)
            if encoder is None:
                return data

            # Ensure values are within valid range
            values = np.clip(
                data_filled.astype(int),
                0,
                len(encoding_info['categories']) - 1
            )

            return pd.Series(
                encoder.inverse_transform(values),
                name=data.name
            )

        elif encoding_info['method'] == 'onehot':
            # Reconstruct from one-hot columns
            columns = encoding_info.get('columns', [])
            if not columns:
                return data

            max_idx = data_filled.astype(int)
            max_idx = np.clip(max_idx, 0, len(columns) - 1)
            return pd.Series(
                [columns[i].split('_')[-1] for i in max_idx],
                name=data.name
            )

        return data