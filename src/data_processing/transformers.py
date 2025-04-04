import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

class DataTransformer:
    """Handles data transformations for different column types with optimized performance"""

    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.encoding_maps = {}  # Store encoding information for inverse transform
        self.data_ranges = {}  # Store original data ranges for validation
        self.missing_flags = {}  # Store information about missing values
        self._cached_transforms = {}  # Cache for transformed columns to improve performance

    def transform_continuous(self, data: pd.Series, method: str = 'standard') -> pd.Series:
        """Transform continuous data using specified method with optimized performance"""
        if data is None:
            raise ValueError(f"None data provided for column transformation")
            
        if data.empty:
            raise ValueError(f"Empty data series provided for column {data.name}")
            
        # Check cache first to avoid redundant transformations
        cache_key = f"{data.name}_{method}_continuous"
        if cache_key in self._cached_transforms:
            return self._cached_transforms[cache_key]

        # Create missing value flag if needed
        has_missing = data.isnull().any()
        if has_missing:
            self.missing_flags[data.name] = data.isnull().astype(int)

            # Create and fit imputer (mean imputation for continuous)
            imputer = SimpleImputer(strategy='mean')
            # Ensure data is reshaped properly and not empty
            valid_data = data.values.reshape(-1, 1)
            if len(valid_data) == 0:
                raise ValueError(f"No valid data after reshaping for column {data.name}")

            imputed_data = imputer.fit_transform(valid_data).flatten()
            self.imputers[data.name] = imputer

            # Use imputed data for further transformations
            data_for_transform = pd.Series(imputed_data, name=data.name)
        else:
            data_for_transform = data

        # Verify data is not empty after transformation
        if data_for_transform.empty:
            raise ValueError(f"No data available for transformation in column {data.name}")

        # Store original data range for this column
        self.data_ranges[data.name] = {
            'min': data_for_transform.min(),
            'max': data_for_transform.max(),
            'dtype': data.dtype,
            'has_missing': has_missing
        }

        # Optimize scaler selection with pre-computed statistics where possible
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))  # Use (-1,1) for GAN compatibility
        elif method == 'standard':
            # If we have more than 100,000 rows, use fixed statistics to avoid full data scan
            if len(data_for_transform) > 100000:
                # Calculate statistics on a sample of data for large datasets
                sample = data_for_transform.sample(n=min(50000, len(data_for_transform)), random_state=42)
                mean = sample.mean()
                var = sample.var()
                scaler = StandardScaler()
                # Set pre-computed statistics
                scaler.mean_ = np.array([mean])
                scaler.var_ = np.array([var])
                scaler.scale_ = np.sqrt(scaler.var_)
                # Do partial_fit with the full dataset
                shaped_data = data_for_transform.values.reshape(-1, 1)
                scaler.partial_fit(shaped_data)
            else:
                scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scalers[data.name] = scaler

        # Ensure data is properly shaped for transformation
        shaped_data = data_for_transform.values.reshape(-1, 1)
        if len(shaped_data) == 0:
            raise ValueError(f"No data available after reshaping in column {data.name}")

        # Use NumPy for faster transformations with large datasets
        if method == 'standard' and len(shaped_data) > 100000 and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            # Manual standardization for large datasets
            transformed_data = (shaped_data - scaler.mean_) / scaler.scale_
            transformed_data = transformed_data.flatten()
        else:
            transformed_data = scaler.fit_transform(shaped_data).flatten()
            
        # Store in cache for future use
        result = pd.Series(transformed_data, name=data.name)
        self._cached_transforms[cache_key] = result
        return result

    def transform_categorical(self, data: pd.Series, method: str = 'label') -> pd.Series:
        """Transform categorical data using specified method with missing value handling"""
        if data is None:
            raise ValueError(f"None data provided for column transformation")
            
        # Check cache first to avoid redundant transformations
        cache_key = f"{data.name}_{method}_categorical"
        if cache_key in self._cached_transforms:
            return self._cached_transforms[cache_key]
            
        # Create missing value flag if needed
        has_missing = data.isnull().any()
        if has_missing:
            self.missing_flags[data.name] = data.isnull().astype(int)
            
            # Create and fit imputer (most frequent imputation for categorical)
            imputer = SimpleImputer(strategy='most_frequent')
            imputed_data = imputer.fit_transform(data.values.reshape(-1, 1)).flatten()
            self.imputers[data.name] = imputer
            
            # Use imputed data for further transformations
            data_for_transform = pd.Series(imputed_data, name=data.name)
        else:
            data_for_transform = data
        
        # Optimization: Calculate category counts only once and reuse
        if len(data_for_transform) > 0:
            # Use more efficient methods for large datasets
            if len(data_for_transform) > 50000:
                # For large datasets, use pandas value_counts which is optimized
                value_counts = data_for_transform.value_counts()
                categories = value_counts.index.tolist()
                category_counts = {cat: count for cat, count in zip(value_counts.index, value_counts.values)}
            else:
                # For smaller datasets, use the regular unique count
                categories = data_for_transform.unique()
                category_counts = {cat: (data_for_transform == cat).sum() for cat in categories}
        else:
            categories = []
            category_counts = {}
            
        if method == 'label':
            encoder = LabelEncoder()
            # For large datasets with many categories, fit on a reduced set
            if len(data_for_transform) > 100000 and len(categories) > 1000:
                # Keep categories that appear at least 0.01% of the time
                min_count = max(1, int(len(data_for_transform) * 0.0001))
                common_categories = [cat for cat, count in category_counts.items() if count >= min_count]
                
                # Map rare categories to a single 'other' category for efficiency
                if len(common_categories) < len(categories):
                    temp_data = data_for_transform.copy()
                    rare_mask = ~data_for_transform.isin(common_categories)
                    temp_data[rare_mask] = 'other'
                    data_for_transform = temp_data
                    # Update categories
                    categories = common_categories + ['other']
            
            self.encoders[data.name] = encoder
            transformed = pd.Series(encoder.fit_transform(data_for_transform), name=data.name)
            
            self.encoding_maps[data.name] = {
                'method': 'label',
                'categories': encoder.classes_,
                'num_categories': len(encoder.classes_),
                'has_missing': has_missing
            }
            
            # Store in cache for future use
            self._cached_transforms[cache_key] = transformed
            return transformed
            
        elif method == 'onehot':
            # For large datasets with many categories, limit the number of dummies
            if len(data_for_transform) > 50000 and len(categories) > 50:
                # Keep top categories that cover at least 95% of data
                sorted_cats = sorted([(cat, count) for cat, count in category_counts.items()], 
                                     key=lambda x: x[1], reverse=True)
                total_count = sum(count for _, count in sorted_cats)
                cumsum = 0
                top_categories = []
                
                for cat, count in sorted_cats:
                    cumsum += count
                    top_categories.append(cat)
                    if cumsum / total_count >= 0.95 or len(top_categories) >= 50:
                        break
                
                # Map less common categories to 'other'
                if len(top_categories) < len(categories):
                    temp_data = data_for_transform.copy()
                    rare_mask = ~data_for_transform.isin(top_categories)
                    temp_data[rare_mask] = 'other'
                    data_for_transform = temp_data
                    # Add 'other' to the list if it's not already there
                    if 'other' not in top_categories:
                        top_categories.append('other')
                    
            # Use a more efficient method for one-hot encoding
            # Apply factorize instead of get_dummies for large datasets
            codes, uniques = data_for_transform.factorize()
            
            self.encoding_maps[data.name] = {
                'method': 'onehot',
                'original_categories': uniques,
                'num_categories': len(uniques),
                'has_missing': has_missing
            }
            
            result = pd.Series(codes, name=data.name)
            # Store in cache for future use
            self._cached_transforms[cache_key] = result
            return result
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def inverse_transform_continuous(self, data: pd.Series) -> pd.Series:
        """Inverse transform continuous data with range validation"""
        scaler = self.scalers.get(data.name)
        if scaler is None:
            # Instead of raising an error, return the original data
            # This handles excluded columns that might be reintroduced
            return data
            
        # Check for None values and handle them
        if data.isnull().any():
            # Fill None values with 0 before transformation
            data = data.fillna(0)

        # Inverse transform
        transformed_data = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()

        # Get original data range
        data_range = self.data_ranges.get(data.name)
        if data_range:
            # Clamp values to original range, handling potential None values
            min_val = data_range.get('min')
            max_val = data_range.get('max')
            
            if min_val is not None and max_val is not None:
                transformed_data = np.clip(
                    transformed_data,
                    min_val,
                    max_val
                )

            # Convert to original dtype if needed
            if data_range.get('dtype') in [np.int32, np.int64, int]:
                transformed_data = np.round(transformed_data).astype(data_range['dtype'])

        return pd.Series(transformed_data, name=data.name)

    def inverse_transform_categorical(self, data: pd.Series) -> pd.Series:
        """Inverse transform categorical data"""
        # Handle None values in the data
        if data.isnull().all():
            return data
            
        encoding_info = self.encoding_maps.get(data.name)
        if encoding_info is None:
            # Instead of raising an error, return the original data
            # This handles excluded columns that might be reintroduced
            return data

        # Fill any None values with 0 before transformation
        data_filled = data.fillna(0)

        if encoding_info['method'] == 'label':
            encoder = self.encoders.get(data.name)
            if encoder is None:
                return data_filled

            # Ensure values are within valid range
            values = np.clip(
                np.round(data_filled.values),
                0,
                len(encoding_info['categories']) - 1
            ).astype(int)

            return pd.Series(
                encoder.inverse_transform(values),
                name=data.name
            )
        elif encoding_info['method'] == 'onehot':
            values = np.clip(
                np.round(data_filled.values),
                0,
                encoding_info['num_categories'] - 1
            ).astype(int)

            return pd.Series(
                encoding_info['original_categories'][values],
                name=data.name
            )

        # For unknown methods, return the original data instead of raising an error
        return data

    def transform_datetime(self, data: pd.Series) -> pd.DataFrame:
        """Transform datetime into multiple features"""
        dt_series = pd.to_datetime(data)
        return pd.DataFrame({
            f"{data.name}_year": dt_series.dt.year,
            f"{data.name}_month": dt_series.dt.month,
            f"{data.name}_day": dt_series.dt.day,
            f"{data.name}_dayofweek": dt_series.dt.dayofweek
        })