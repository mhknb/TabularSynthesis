"""
Adapter module for table-evaluator with strict type enforcement
"""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class TableEvaluatorAdapter:
    """Adapter class to integrate table-evaluator functionality"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cat_cols: Optional[List[str]] = None):
        """Initialize with real and synthetic data"""
        try:
            print("\n=== TableEvaluatorAdapter Initialization ===")
            self._log_dataframe_info("Initial real data", real_data)
            self._log_dataframe_info("Initial synthetic data", synthetic_data)

            self.real_data = real_data.copy()
            self.synthetic_data = synthetic_data.copy()

            # Process data
            self._initialize_evaluator(cat_cols)

        except Exception as e:
            print(f"\nFATAL ERROR in initialization: {str(e)}")
            self._log_detailed_error(e)
            raise

    def _log_dataframe_info(self, name: str, df: pd.DataFrame):
        """Log detailed DataFrame information"""
        print(f"\n{name}:")
        print(f"Shape: {df.shape}")
        print("Data Types:")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
            print(f"Sample values: {df[col].head().tolist()}")

    def _log_detailed_error(self, error: Exception):
        """Log detailed error information"""
        print("\n=== Error Details ===")
        print(f"Error Type: {type(error).__name__}")
        print(f"Error Message: {str(error)}")
        if hasattr(self, 'real_processed') and hasattr(self, 'synthetic_processed'):
            print("\nProcessed Data State at Error:")
            for col in self.real_processed.columns:
                print(f"\nColumn: {col}")
                print(f"Real dtype: {self.real_processed[col].dtype}")
                print(f"Real sample: {self.real_processed[col].head().tolist()}")
                print(f"Synthetic dtype: {self.synthetic_processed[col].dtype}")
                print(f"Synthetic sample: {self.synthetic_processed[col].head().tolist()}")

    def _force_numpy_float64(self, series: pd.Series, col_name: str) -> np.ndarray:
        """Force convert a series to numpy float64 array with strict type checking"""
        try:
            # Convert to string first to handle mixed types
            str_series = series.astype(str)

            # Try converting to float
            float_array = pd.to_numeric(str_series, errors='coerce').fillna(0)

            # Force numpy float64
            result = float_array.to_numpy(dtype=np.float64)

            if not isinstance(result.dtype, np.dtype) or result.dtype != np.float64:
                raise TypeError(f"Failed to convert {col_name} to numpy.float64")

            return result
        except Exception as e:
            print(f"Error converting {col_name} to numpy.float64:")
            print(f"Original dtype: {series.dtype}")
            print(f"Sample values: {series.head().tolist()}")
            raise

    def _initialize_evaluator(self, cat_cols: Optional[List[str]] = None):
        """Initialize table-evaluator with properly processed data"""
        try:
            # Identify categorical columns
            self.cat_cols = cat_cols or self._infer_categorical_columns()
            print(f"\nCategorical columns: {self.cat_cols}")

            # Initialize processed DataFrames
            self.real_processed = pd.DataFrame()
            self.synthetic_processed = pd.DataFrame()

            # Process each column separately
            for col in self.real_data.columns:
                print(f"\nProcessing column: {col}")

                if col in self.cat_cols:
                    # Handle categorical
                    encoder = LabelEncoder()
                    all_values = pd.concat([
                        self.real_data[col].astype(str),
                        self.synthetic_data[col].astype(str)
                    ]).unique()
                    encoder.fit(all_values)

                    # Transform to numeric arrays
                    real_encoded = self._force_numpy_float64(
                        pd.Series(encoder.transform(self.real_data[col].astype(str))),
                        f"{col} (real)"
                    )
                    synth_encoded = self._force_numpy_float64(
                        pd.Series(encoder.transform(self.synthetic_data[col].astype(str))),
                        f"{col} (synthetic)"
                    )
                else:
                    # Handle numeric
                    real_encoded = self._force_numpy_float64(
                        self.real_data[col],
                        f"{col} (real)"
                    )
                    synth_encoded = self._force_numpy_float64(
                        self.synthetic_data[col],
                        f"{col} (synthetic)"
                    )

                # Add to processed DataFrames
                self.real_processed[col] = real_encoded
                self.synthetic_processed[col] = synth_encoded

                # Verify types
                print(f"Processed {col}:")
                print(f"Real dtype: {self.real_processed[col].dtype}")
                print(f"Synthetic dtype: {self.synthetic_processed[col].dtype}")

            # Final verification
            print("\nVerifying all columns are float64:")
            for col in self.real_processed.columns:
                if not np.issubdtype(self.real_processed[col].dtype, np.float64):
                    raise TypeError(f"Column {col} (real) is not float64: {self.real_processed[col].dtype}")
                if not np.issubdtype(self.synthetic_processed[col].dtype, np.float64):
                    raise TypeError(f"Column {col} (synthetic) is not float64: {self.synthetic_processed[col].dtype}")

            # Initialize evaluator
            print("\nInitializing TableEvaluator...")
            self.evaluator = TableEvaluator(
                real=self.real_processed,
                fake=self.synthetic_processed,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")

        except Exception as e:
            print(f"\nError in initialization: {str(e)}")
            self._log_detailed_error(e)
            raise

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data types and unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            if pd.api.types.is_object_dtype(self.real_data[col]) or pd.api.types.is_categorical_dtype(self.real_data[col]):
                categorical_columns.append(col)
            else:
                n_unique = self.real_data[col].nunique()
                if n_unique < 20:  # Consider low cardinality columns as categorical
                    categorical_columns.append(col)
        return categorical_columns

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        try:
            print(f"\nStarting evaluation with target column: {target_col}")

            # Verify target column
            if target_col not in self.real_processed.columns:
                raise ValueError(f"Target column {target_col} not found")

            print(f"Target column type - Real: {self.real_processed[target_col].dtype}")
            print(f"Target column type - Synthetic: {self.synthetic_processed[target_col].dtype}")

            # Run evaluation
            ml_scores = self.evaluator.evaluate(target_col=target_col)
            print("Evaluation completed successfully")

            return {
                'classifier_scores': ml_scores.get('Classifier F1-scores', None),
                'privacy': {
                    'Duplicate rows between sets (real/fake)': ml_scores.get('Duplicate rows between sets (real/fake)', (0, 0)),
                    'nearest neighbor mean': ml_scores.get('nearest neighbor mean', 0),
                    'nearest neighbor std': ml_scores.get('nearest neighbor std', 0)
                },
                'correlation': {
                    'Column Correlation Distance RMSE': ml_scores.get('Column Correlation Distance RMSE', 0),
                    'Column Correlation distance MAE': ml_scores.get('Column Correlation distance MAE', 0)
                },
                'similarity': {
                    'basic statistics': ml_scores.get('basic statistics', 0),
                    'Correlation column correlations': ml_scores.get('Correlation column correlations', 0),
                    'Mean Correlation between fake and real columns': ml_scores.get('Mean Correlation between fake and real columns', 0),
                    '1 - MAPE Estimator results': ml_scores.get('1 - MAPE Estimator results', 0),
                    '1 - MAPE 5 PCA components': ml_scores.get('1 - MAPE 5 PCA components', 0),
                    'Similarity Score': ml_scores.get('Similarity Score', 0)
                }
            }

        except Exception as e:
            print(f"\nError in evaluation: {str(e)}")
            self._log_detailed_error(e)
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            return self.evaluator.visual_evaluation()
        except Exception as e:
            print(f"\nError in visual evaluation: {str(e)}")
            self._log_detailed_error(e)
            raise