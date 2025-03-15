"""
Adapter module for table-evaluator with comprehensive type checking
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
        self._validate_input_data(real_data, synthetic_data)

        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Process and validate data
        self._initialize_evaluator(cat_cols)

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

    def _initialize_evaluator(self, cat_cols: Optional[List[str]] = None):
        """Initialize table-evaluator with properly processed data"""
        try:
            # 1. Identify categorical columns
            self.cat_cols = cat_cols or self._infer_categorical_columns()
            print(f"\nCategorical columns: {self.cat_cols}")

            # 2. Create processed copies
            self.real_processed = self.real_data.copy()
            self.synthetic_processed = self.synthetic_data.copy()

            # 3. Process categorical columns
            self._process_categorical_columns()

            # 4. Process numerical columns
            self._process_numerical_columns()

            # 5. Final validation
            self._validate_processed_data()

            # 6. Initialize evaluator
            self.evaluator = TableEvaluator(
                real=self.real_processed,
                fake=self.synthetic_processed,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")

        except Exception as e:
            print(f"\nError in initialization: {str(e)}")
            self._print_debug_info()
            raise

    def _process_categorical_columns(self):
        """Process categorical columns ensuring float64 output"""
        print("\nProcessing categorical columns:")

        for col in self.cat_cols:
            try:
                print(f"\nProcessing {col}:")
                # Get all unique values
                all_values = pd.concat([
                    self.real_processed[col],
                    self.synthetic_processed[col]
                ]).astype(str).unique()

                # Encode values
                encoder = LabelEncoder()
                encoder.fit(all_values)

                # Transform and convert to float64
                self.real_processed[col] = encoder.transform(
                    self.real_processed[col].astype(str)
                ).astype(np.float64)

                self.synthetic_processed[col] = encoder.transform(
                    self.synthetic_processed[col].astype(str)
                ).astype(np.float64)

                print(f"Processed {col} - dtype: {self.real_processed[col].dtype}")

            except Exception as e:
                print(f"Error processing categorical column {col}: {str(e)}")
                print(f"Values sample: {self.real_processed[col].head()}")
                raise

    def _process_numerical_columns(self):
        """Process numerical columns ensuring float64 output"""
        print("\nProcessing numerical columns:")

        numerical_cols = [col for col in self.real_processed.columns 
                         if col not in self.cat_cols]

        for col in numerical_cols:
            try:
                print(f"\nProcessing {col}:")
                # Convert to numeric
                self.real_processed[col] = pd.to_numeric(
                    self.real_processed[col], 
                    errors='coerce'
                ).fillna(0.0).astype(np.float64)

                self.synthetic_processed[col] = pd.to_numeric(
                    self.synthetic_processed[col], 
                    errors='coerce'
                ).fillna(0.0).astype(np.float64)

                # Scale to [0,1] for numerical stability
                scaler = MinMaxScaler()
                self.real_processed[col] = scaler.fit_transform(
                    self.real_processed[col].values.reshape(-1, 1)
                ).ravel()

                self.synthetic_processed[col] = scaler.transform(
                    self.synthetic_processed[col].values.reshape(-1, 1)
                ).ravel()

                print(f"Processed {col} - dtype: {self.real_processed[col].dtype}")

            except Exception as e:
                print(f"Error processing numerical column {col}: {str(e)}")
                print(f"Values sample: {self.real_processed[col].head()}")
                raise

    def _validate_processed_data(self):
        """Validate processed data before evaluation"""
        print("\nValidating processed data:")

        # Check dtypes
        print("\nProcessed data types:")
        print("Real data:\n", self.real_processed.dtypes)
        print("\nSynthetic data:\n", self.synthetic_processed.dtypes)

        # Verify no object types remain
        for df_name, df in [("Real", self.real_processed), 
                          ("Synthetic", self.synthetic_processed)]:
            obj_cols = df.select_dtypes(include=['object']).columns
            if not obj_cols.empty:
                raise ValueError(f"{df_name} data contains object columns: {obj_cols.tolist()}")

        # Verify all numeric
        for df_name, df in [("Real", self.real_processed), 
                          ("Synthetic", self.synthetic_processed)]:
            non_float = df.select_dtypes(exclude=[np.float64]).columns
            if not non_float.empty:
                raise ValueError(f"{df_name} data contains non-float64 columns: {non_float.tolist()}")

    def _print_debug_info(self):
        """Print debug information for troubleshooting"""
        print("\nDebug Information:")
        for col in self.real_processed.columns:
            print(f"\nColumn: {col}")
            print(f"Real data type: {self.real_processed[col].dtype}")
            print(f"Synthetic data type: {self.synthetic_processed[col].dtype}")
            print(f"Real sample: {self.real_processed[col].head()}")
            print(f"Synthetic sample: {self.synthetic_processed[col].head()}")

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data types and unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            if self.real_data[col].dtype in ['object', 'category']:
                categorical_columns.append(col)
            else:
                n_unique = self.real_data[col].nunique()
                if n_unique < 20:  # Lower threshold for categorical data
                    categorical_columns.append(col)
        return categorical_columns

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        try:
            print(f"\nStarting evaluation with target column: {target_col}")

            # Verify target column
            if target_col not in self.real_processed.columns:
                raise ValueError(f"Target column {target_col} not found in processed data")

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
            self._print_debug_info()
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            return self.evaluator.visual_evaluation()
        except Exception as e:
            print(f"Error in visual evaluation: {str(e)}")
            self._print_debug_info()
            raise