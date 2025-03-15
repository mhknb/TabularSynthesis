"""
Adapter module for table-evaluator with enhanced error tracking
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
            print("Input Data Types:")
            print("Real data:\n", real_data.dtypes)
            print("\nSynthetic data:\n", synthetic_data.dtypes)

            self.real_data = real_data.copy()
            self.synthetic_data = synthetic_data.copy()

            # Process data
            self._initialize_evaluator(cat_cols)

        except Exception as e:
            print(f"\nFATAL ERROR in initialization: {str(e)}")
            print("Stack trace:", e.__traceback__)
            raise

    def _initialize_evaluator(self, cat_cols: Optional[List[str]] = None):
        """Initialize table-evaluator with properly processed data"""
        try:
            print("\n=== Processing Steps ===")

            # 1. Identify columns
            self.cat_cols = cat_cols or self._infer_categorical_columns()
            print(f"\nCategorical columns: {self.cat_cols}")

            # 2. Create working copies
            self.real_processed = self.real_data.copy()
            self.synthetic_processed = self.synthetic_data.copy()

            # 3. Convert all data to float64
            self._convert_all_to_float64()

            # 4. Initialize evaluator
            print("\nInitializing TableEvaluator...")
            self.evaluator = TableEvaluator(
                real=self.real_processed,
                fake=self.synthetic_processed,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")

        except Exception as e:
            print(f"\nERROR in initialization: {str(e)}")
            self._print_debug_info()
            raise

    def _convert_all_to_float64(self):
        """Convert all columns to float64 with detailed error tracking"""
        try:
            print("\n=== Converting Data Types ===")

            for col in self.real_processed.columns:
                print(f"\nProcessing column: {col}")
                try:
                    # Check initial types
                    print(f"Initial types - Real: {self.real_processed[col].dtype}, Synthetic: {self.synthetic_processed[col].dtype}")

                    if col in self.cat_cols:
                        self._convert_categorical_to_float64(col)
                    else:
                        self._convert_numeric_to_float64(col)

                    # Verify conversion
                    if not isinstance(self.real_processed[col].dtype, np.float64().dtype.__class__):
                        raise TypeError(f"Column {col} not converted to float64")

                    print(f"Final types - Real: {self.real_processed[col].dtype}, Synthetic: {self.synthetic_processed[col].dtype}")

                except Exception as col_error:
                    print(f"Error processing column {col}: {str(col_error)}")
                    print(f"Sample values - Real: {self.real_processed[col].head()}")
                    print(f"Sample values - Synthetic: {self.synthetic_processed[col].head()}")
                    raise

        except Exception as e:
            print(f"ERROR in type conversion: {str(e)}")
            raise

    def _convert_categorical_to_float64(self, col: str):
        """Convert categorical column to float64"""
        try:
            print(f"Converting categorical column {col}")

            # Convert to string first
            real_vals = self.real_processed[col].astype(str)
            synth_vals = self.synthetic_processed[col].astype(str)

            # Encode to numeric
            encoder = LabelEncoder()
            all_values = pd.concat([real_vals, synth_vals]).unique()
            encoder.fit(all_values)

            # Transform and explicitly convert to float64
            self.real_processed[col] = encoder.transform(real_vals).astype(np.float64)
            self.synthetic_processed[col] = encoder.transform(synth_vals).astype(np.float64)

            print(f"Converted {col} - Real type: {self.real_processed[col].dtype}")

        except Exception as e:
            print(f"ERROR converting categorical column {col}: {str(e)}")
            raise

    def _convert_numeric_to_float64(self, col: str):
        """Convert numeric column to float64"""
        try:
            print(f"Converting numeric column {col}")

            # Convert to numeric with NaN handling
            self.real_processed[col] = pd.to_numeric(self.real_processed[col], errors='coerce').fillna(0.0)
            self.synthetic_processed[col] = pd.to_numeric(self.synthetic_processed[col], errors='coerce').fillna(0.0)

            # Scale values
            scaler = MinMaxScaler()
            self.real_processed[col] = scaler.fit_transform(
                self.real_processed[col].values.reshape(-1, 1)
            ).ravel().astype(np.float64)

            self.synthetic_processed[col] = scaler.transform(
                self.synthetic_processed[col].values.reshape(-1, 1)
            ).ravel().astype(np.float64)

            print(f"Converted {col} - Real type: {self.real_processed[col].dtype}")

        except Exception as e:
            print(f"ERROR converting numeric column {col}: {str(e)}")
            raise

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data types and unique values"""
        categorical_columns = []
        try:
            for col in self.real_data.columns:
                if self.real_data[col].dtype in ['object', 'category']:
                    categorical_columns.append(col)
                else:
                    n_unique = self.real_data[col].nunique()
                    if n_unique < 20:  # Consider low cardinality columns as categorical
                        categorical_columns.append(col)

            print(f"Inferred categorical columns: {categorical_columns}")
            return categorical_columns

        except Exception as e:
            print(f"ERROR inferring categorical columns: {str(e)}")
            raise

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        try:
            print(f"\n=== Starting Evaluation ===")
            print(f"Target column: {target_col}")
            print(f"Target column types - Real: {self.real_processed[target_col].dtype}")
            print(f"Target column types - Synthetic: {self.synthetic_processed[target_col].dtype}")

            # Verify target column
            if target_col not in self.real_processed.columns:
                raise ValueError(f"Target column {target_col} not found")

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
            print(f"\nERROR in evaluation: {str(e)}")
            self._print_debug_info()
            raise

    def _print_debug_info(self):
        """Print detailed debug information"""
        print("\n=== Debug Information ===")
        try:
            print("\nDataFrame Information:")
            print("Real processed shape:", self.real_processed.shape)
            print("Synthetic processed shape:", self.synthetic_processed.shape)

            print("\nColumn Types:")
            print("Real processed types:\n", self.real_processed.dtypes)
            print("\nSynthetic processed types:\n", self.synthetic_processed.dtypes)

            print("\nSample Values:")
            for col in self.real_processed.columns:
                print(f"\nColumn: {col}")
                print(f"Real samples: {self.real_processed[col].head()}")
                print(f"Synthetic samples: {self.synthetic_processed[col].head()}")

        except Exception as e:
            print(f"ERROR in debug info: {str(e)}")

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            return self.evaluator.visual_evaluation()
        except Exception as e:
            print(f"\nERROR in visual evaluation: {str(e)}")
            self._print_debug_info()
            raise

    def _validate_input_data(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Validate input data and print detailed information"""
        print("\nValidating input data:")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print("\nReal data types:")
        print(real_data.dtypes)
        print("\nSynthetic data types:")
        print(synthetic_data.dtypes)

        # Check for common columns
        common_cols = set(real_data.columns) & set(synthetic_data.columns)
        if not common_cols:
            raise ValueError("No common columns found between real and synthetic data")
        print(f"\nCommon columns: {common_cols}")