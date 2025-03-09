import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

class DataTransformer:
    """Handles data transformations for different column types"""

    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.encoding_maps = {}  
        self.data_ranges = {}  
        self.missing_flags = {}  

    def transform_continuous(self, data: pd.Series, method: str = 'minmax') -> pd.Series:
        """Transform continuous data using specified method with missing value handling"""
        # Create missing value flag if needed
        has_missing = data.isnull().any()
        if has_missing:
            self.missing_flags[data.name] = data.isnull().astype(int)

            # Create and fit imputer (mean imputation for continuous)
            imputer = SimpleImputer(strategy='mean')
            imputed_data = imputer.fit_transform(data.values.reshape(-1, 1)).flatten()
            self.imputers[data.name] = imputer

            # Use imputed data for further transformations
            data_for_transform = pd.Series(imputed_data, name=data.name)
        else:
            data_for_transform = data

        # Store original data range and characteristics
        self.data_ranges[data.name] = {
            'min': float(data_for_transform.min()),
            'max': float(data_for_transform.max()),
            'dtype': str(data.dtype),
            'has_missing': has_missing,
            'is_large_number': data_for_transform.abs().max() > 1e6
        }

        # Apply log transformation for large numbers (e.g., Volume)
        if self.data_ranges[data.name]['is_large_number']:
            # Shift data to be positive for log transform
            min_val = data_for_transform.min()
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            data_for_transform = np.log1p(data_for_transform + shift)
            self.data_ranges[data.name]['shift'] = shift

        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scalers[data.name] = scaler
        transformed = scaler.fit_transform(data_for_transform.values.reshape(-1, 1)).flatten()
        return pd.Series(transformed, name=data.name)

    def inverse_transform_continuous(self, data: pd.Series) -> pd.Series:
        """Inverse transform continuous data with improved range validation"""
        scaler = self.scalers.get(data.name)
        if scaler is None:
            raise ValueError(f"No scaler found for column {data.name}")

        # Get original data range
        data_range = self.data_ranges.get(data.name)
        if not data_range:
            raise ValueError(f"No data range information found for column {data.name}")

        # Inverse transform scaling
        transformed_data = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()

        # Inverse log transform if it was a large number
        if data_range.get('is_large_number', False):
            transformed_data = np.expm1(transformed_data)
            if 'shift' in data_range:
                transformed_data -= data_range['shift']

        # Ensure values are within reasonable bounds
        transformed_data = np.clip(
            transformed_data,
            data_range['min'],
            data_range['max']
        )

        # Handle dtype conversion carefully
        if 'int' in data_range['dtype']:
            transformed_data = np.round(transformed_data)
            # Ensure we don't exceed the dtype's limits
            if 'int64' in data_range['dtype']:
                transformed_data = np.clip(
                    transformed_data,
                    np.iinfo(np.int64).min,
                    np.iinfo(np.int64).max
                )
            transformed_data = transformed_data.astype(data_range['dtype'])

        return pd.Series(transformed_data, name=data.name)

    def transform_categorical(self, data: pd.Series, method: str = 'label') -> pd.Series:
        """Transform categorical data using specified method with missing value handling"""
        has_missing = data.isnull().any()
        if has_missing:
            self.missing_flags[data.name] = data.isnull().astype(int)

            imputer = SimpleImputer(strategy='most_frequent')
            imputed_data = imputer.fit_transform(data.values.reshape(-1, 1)).flatten()
            self.imputers[data.name] = imputer

            data_for_transform = pd.Series(imputed_data, name=data.name)
        else:
            data_for_transform = data

        if method == 'label':
            encoder = LabelEncoder()
            self.encoders[data.name] = encoder
            transformed = pd.Series(encoder.fit_transform(data_for_transform), name=data.name)
            self.encoding_maps[data.name] = {
                'method': 'label',
                'categories': encoder.classes_,
                'num_categories': len(encoder.classes_),
                'has_missing': has_missing
            }
            return transformed
        elif method == 'onehot':
            dummies = pd.get_dummies(data_for_transform, prefix=data.name)
            self.encoding_maps[data.name] = {
                'method': 'onehot',
                'columns': list(dummies.columns),
                'original_categories': data_for_transform.unique(),
                'num_categories': len(data_for_transform.unique()),
                'has_missing': has_missing
            }
            return pd.Series(data_for_transform.factorize()[0], name=data.name)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def inverse_transform_categorical(self, data: pd.Series) -> pd.Series:
        """Inverse transform categorical data"""
        encoding_info = self.encoding_maps.get(data.name)
        if encoding_info is None:
            raise ValueError(f"No encoding information found for column {data.name}")

        if encoding_info['method'] == 'label':
            encoder = self.encoders.get(data.name)
            if encoder is None:
                raise ValueError(f"No encoder found for column {data.name}")

            # Ensure values are within valid range
            values = np.clip(
                np.round(data.values),
                0,
                len(encoding_info['categories']) - 1
            ).astype(int)

            return pd.Series(
                encoder.inverse_transform(values),
                name=data.name
            )
        elif encoding_info['method'] == 'onehot':
            values = np.clip(
                np.round(data.values),
                0,
                encoding_info['num_categories'] - 1
            ).astype(int)

            return pd.Series(
                encoding_info['original_categories'][values],
                name=data.name
            )

        raise ValueError(f"Unknown encoding method for column {data.name}")

    def transform_datetime(self, data: pd.Series) -> pd.DataFrame:
        """Transform datetime into multiple features"""
        dt_series = pd.to_datetime(data)
        return pd.DataFrame({
            f"{data.name}_year": dt_series.dt.year,
            f"{data.name}_month": dt_series.dt.month,
            f"{data.name}_day": dt_series.dt.day,
            f"{data.name}_dayofweek": dt_series.dt.dayofweek
        })