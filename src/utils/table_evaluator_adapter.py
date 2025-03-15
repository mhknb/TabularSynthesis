"""
Adapter module for integrating table-evaluator functionality
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator

class TableEvaluatorAdapter:
    """Adapter class to integrate table-evaluator functionality"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cat_cols: Optional[List[str]] = None):
        """Initialize the adapter with real and synthetic datasets"""
        print("\nDEBUG - TableEvaluatorAdapter initialization:")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")

        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.cat_cols = cat_cols or self._infer_categorical_columns()

        print("\nDEBUG - Initial data types:")
        for col in self.real_data.columns:
            print(f"Column '{col}':")
            print(f"  Real data type: {self.real_data[col].dtype}")
            print(f"  Synthetic data type: {self.synthetic_data[col].dtype}")
            print(f"  Sample real values: {self.real_data[col].head().tolist()}")

        # Convert all data to numeric format
        self._convert_to_numeric()

        print("\nDEBUG - Final data types after conversion:")
        for col in self.real_data.columns:
            print(f"Column '{col}':")
            print(f"  Real data type: {self.real_data[col].dtype}")
            print(f"  Synthetic data type: {self.synthetic_data[col].dtype}")

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data type and unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            # Check data type first
            if pd.api.types.is_object_dtype(self.real_data[col]) or \
               pd.api.types.is_categorical_dtype(self.real_data[col]):
                categorical_columns.append(col)
            else:
                # Check number of unique values
                n_unique = self.real_data[col].nunique()
                if n_unique < 50:  # Consider columns with less than 50 unique values as categorical
                    categorical_columns.append(col)

        print(f"Inferred categorical columns: {categorical_columns}")
        return categorical_columns

    def _convert_to_numeric(self):
        """Convert all columns to numeric format"""
        print("\nConverting all columns to numeric format...")

        for col in self.real_data.columns:
            print(f"\nProcessing column: {col}")

            try:
                if col in self.cat_cols:
                    # Convert categorical to numeric codes
                    print(f"Converting categorical column {col} to numeric codes")
                    # Combine unique values from both datasets
                    unique_values = pd.concat([
                        self.real_data[col],
                        self.synthetic_data[col]
                    ]).unique()

                    # Create mapping dictionary
                    value_map = {val: idx for idx, val in enumerate(unique_values)}

                    # Convert to numeric using mapping
                    self.real_data[col] = self.real_data[col].map(value_map).astype('float64')
                    self.synthetic_data[col] = self.synthetic_data[col].map(value_map).astype('float64')
                else:
                    # Try converting to float64
                    self.real_data[col] = pd.to_numeric(self.real_data[col], errors='coerce').astype('float64')
                    self.synthetic_data[col] = pd.to_numeric(self.synthetic_data[col], errors='coerce').astype('float64')

                # Fill any NaN values with 0
                self.real_data[col] = self.real_data[col].fillna(0)
                self.synthetic_data[col] = self.synthetic_data[col].fillna(0)

                print(f"Converted {col} - Real dtype: {self.real_data[col].dtype}, Synthetic dtype: {self.synthetic_data[col].dtype}")

            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")
                print("Attempting fallback conversion to numeric codes...")
                # Fallback to treating as categorical
                unique_values = pd.concat([
                    self.real_data[col].astype(str),
                    self.synthetic_data[col].astype(str)
                ]).unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                self.real_data[col] = self.real_data[col].astype(str).map(value_map).astype('float64')
                self.synthetic_data[col] = self.synthetic_data[col].astype(str).map(value_map).astype('float64')
                if col not in self.cat_cols:
                    self.cat_cols.append(col)

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation using table-evaluator"""
        try:
            print(f"\nDEBUG - Starting evaluation for target column: {target_col}")
            print(f"Target column type - Real: {self.real_data[target_col].dtype}")
            print(f"Target column type - Synthetic: {self.synthetic_data[target_col].dtype}")

            # Verify all data is numeric
            for col in self.real_data.columns:
                if not np.issubdtype(self.real_data[col].dtype, np.number):
                    raise ValueError(f"Column {col} in real data is not numeric: {self.real_data[col].dtype}")
                if not np.issubdtype(self.synthetic_data[col].dtype, np.number):
                    raise ValueError(f"Column {col} in synthetic data is not numeric: {self.synthetic_data[col].dtype}")

            # Initialize TableEvaluator
            print("Initializing TableEvaluator...")
            evaluator = TableEvaluator(
                real=self.real_data,
                fake=self.synthetic_data,
                cat_cols=self.cat_cols
            )

            # Run evaluation
            print("Running evaluation...")
            ml_scores = evaluator.evaluate(target_col=target_col)
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
            print("\nDebug information:")
            print("\nData shape:")
            print(f"Real data shape: {self.real_data.shape}")
            print(f"Synthetic data shape: {self.synthetic_data.shape}")
            print("\nColumn dtypes:")
            for col in self.real_data.columns:
                print(f"Column {col}:")
                print(f"  Real dtype: {self.real_data[col].dtype}")
                print(f"  Synthetic dtype: {self.synthetic_data[col].dtype}")
                print(f"  Real sample: {self.real_data[col].head()}")
                print(f"  Synthetic sample: {self.synthetic_data[col].head()}")
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            print("\nDEBUG - Generating visual evaluation")
            evaluator = TableEvaluator(
                real=self.real_data,
                fake=self.synthetic_data,
                cat_cols=self.cat_cols
            )
            plots = evaluator.visual_evaluation()
            print("Visual evaluation completed successfully")
            return plots
        except Exception as e:
            print(f"\nError in visual evaluation: {str(e)}")
            raise

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data type and unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            # Check data type first
            if pd.api.types.is_object_dtype(self.real_data[col]) or \
               pd.api.types.is_categorical_dtype(self.real_data[col]):
                categorical_columns.append(col)
            else:
                # Check number of unique values
                n_unique = self.real_data[col].nunique()
                if n_unique < 50:  # Consider columns with less than 50 unique values as categorical
                    categorical_columns.append(col)

        print(f"Inferred categorical columns: {categorical_columns}")
        return categorical_columns