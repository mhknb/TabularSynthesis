"""
Adapter module for integrating table-evaluator functionality with enhanced logging
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator
from sklearn.preprocessing import LabelEncoder

class TableEvaluatorAdapter:
    """Adapter class to integrate table-evaluator functionality"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cat_cols: Optional[List[str]] = None):
        """Initialize with real and synthetic data"""
        print("\nDEBUG - TableEvaluatorAdapter initialization:")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print(f"Real data columns: {real_data.columns.tolist()}")
        print(f"Synthetic data columns: {synthetic_data.columns.tolist()}")

        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Remove duplicates from synthetic data
        orig_len = len(self.synthetic_data)
        self.synthetic_data = self.synthetic_data.drop_duplicates()
        dropped_dupes = orig_len - len(self.synthetic_data)
        print(f"Removed {dropped_dupes} duplicate rows from synthetic data")

        # Find common columns for evaluation
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        print(f"Common columns for evaluation: {common_cols}")

        if not common_cols:
            raise ValueError("No common columns found between real and synthetic data!")

        # Only use columns that exist in both datasets
        self.real_data = self.real_data[common_cols]
        self.synthetic_data = self.synthetic_data[common_cols]

        # Identify categorical columns
        self.cat_cols = cat_cols or self._infer_categorical_columns()
        print(f"Categorical columns: {self.cat_cols}")

        # Preprocess data
        self.real_processed, self.synthetic_processed = self._preprocess_data()
        print("Data preprocessing completed")

        try:
            print("\nBefore TableEvaluator initialization:")
            print("\nReal data types:")
            print(self.real_processed.dtypes)
            print("\nSynthetic data types:")
            print(self.synthetic_processed.dtypes)

            # Verify no object types remain
            real_objects = self.real_processed.select_dtypes(include=['object']).columns
            synth_objects = self.synthetic_processed.select_dtypes(include=['object']).columns

            if len(real_objects) > 0 or len(synth_objects) > 0:
                print("\nWARNING: Object types found:")
                print(f"Real data object columns: {real_objects.tolist()}")
                print(f"Synthetic data object columns: {synth_objects.tolist()}")

                # Force conversion of any remaining object columns to float
                for col in real_objects:
                    print(f"\nConverting column {col} to numeric:")
                    print(f"Real unique values: {self.real_processed[col].unique()}")
                    self.real_processed[col] = pd.to_numeric(self.real_processed[col], errors='coerce')
                    self.synthetic_processed[col] = pd.to_numeric(self.synthetic_processed[col], errors='coerce')

            self.evaluator = TableEvaluator(
                real=self.real_processed,
                fake=self.synthetic_processed,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")
        except Exception as e:
            print(f"\nError initializing TableEvaluator: {str(e)}")
            print("\nDetailed data inspection:")
            for col in self.real_processed.columns:
                print(f"\nColumn: {col}")
                print(f"Real data type: {self.real_processed[col].dtype}")
                print(f"Synthetic data type: {self.synthetic_processed[col].dtype}")
                print(f"Real unique values: {self.real_processed[col].unique()[:5]}")
                print(f"Synthetic unique values: {self.synthetic_processed[col].unique()[:5]}")
            raise

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data types and unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            if self.real_data[col].dtype in ['object', 'category']:
                categorical_columns.append(col)
            else:
                n_unique = self.real_data[col].nunique()
                if n_unique < 50:  # Consider columns with less than 50 unique values as categorical
                    categorical_columns.append(col)
        return categorical_columns

    def _preprocess_data(self) -> tuple:
        """Preprocess data for evaluation"""
        real_processed = self.real_data.copy()
        synthetic_processed = self.synthetic_data.copy()

        # Handle categorical columns
        label_encoders = {}
        for col in self.cat_cols:
            if col in real_processed.columns:
                try:
                    # Combine unique values from both datasets
                    unique_values = pd.concat([real_processed[col], synthetic_processed[col]]).unique()
                    print(f"\nProcessing categorical column {col}")
                    print(f"Unique values: {unique_values[:5]}")

                    # Create and fit label encoder
                    le = LabelEncoder()
                    le.fit(unique_values)

                    # Transform both datasets
                    real_processed[col] = le.transform(real_processed[col])
                    synthetic_processed[col] = le.transform(synthetic_processed[col])

                    label_encoders[col] = le
                    print(f"Successfully encoded column {col}")
                except Exception as e:
                    print(f"Error encoding column {col}: {str(e)}")
                    raise

        # Handle numeric columns
        numeric_cols = [col for col in real_processed.columns if col not in self.cat_cols]
        for col in numeric_cols:
            try:
                print(f"\nProcessing numeric column {col}")
                # Convert to float and handle missing values
                real_processed[col] = pd.to_numeric(real_processed[col], errors='coerce').fillna(0).astype(float)
                synthetic_processed[col] = pd.to_numeric(synthetic_processed[col], errors='coerce').fillna(0).astype(float)
                print(f"Successfully processed column {col}")
            except Exception as e:
                print(f"Error processing numeric column {col}: {str(e)}")
                raise

        print("\nProcessed data types:")
        print("Real data:\n", real_processed.dtypes)
        print("\nSynthetic data:\n", synthetic_processed.dtypes)

        return real_processed, synthetic_processed

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        try:
            print("\nDEBUG - Running table-evaluator evaluation")
            print(f"Target column: {target_col}")

            # Verify target column type
            print(f"Target column type in real data: {self.real_processed[target_col].dtype}")
            print(f"Target column type in synthetic data: {self.synthetic_processed[target_col].dtype}")

            # Get evaluation results
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
            print(f"Error in evaluation: {str(e)}")
            # Add detailed error context
            print("\nDetailed data state at error:")
            print("Real data types:", self.real_processed.dtypes)
            print("Synthetic data types:", self.synthetic_processed.dtypes)
            print(f"Target column '{target_col}' details:")
            print("Real data unique values:", self.real_processed[target_col].unique()[:5])
            print("Synthetic data unique values:", self.synthetic_processed[target_col].unique()[:5])
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            print("\nDEBUG - Generating visual evaluation")
            plots = self.evaluator.visual_evaluation()
            print("Visual evaluation completed successfully")
            return plots
        except Exception as e:
            print(f"Error in visual evaluation: {str(e)}")
            raise