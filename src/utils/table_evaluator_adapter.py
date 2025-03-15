"""
Adapter module for integrating table-evaluator functionality
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator
from sklearn.preprocessing import LabelEncoder

class TableEvaluatorAdapter:
    """Adapter class to integrate table-evaluator functionality"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cat_cols: Optional[List[str]] = None):
        """
        Initialize the adapter with real and synthetic data

        Args:
            real_data: DataFrame containing real data
            synthetic_data: DataFrame containing synthetic data
            cat_cols: List of categorical column names
        """
        print("\nDEBUG - TableEvaluatorAdapter initialization:")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")

        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Identify categorical columns
        self.cat_cols = cat_cols or self._infer_categorical_columns()
        print(f"Categorical columns: {self.cat_cols}")

        # Preprocess data
        self.real_processed, self.synthetic_processed = self._preprocess_data()
        print("Data preprocessing completed")

        try:
            self.evaluator = TableEvaluator(
                real=self.real_processed,
                fake=self.synthetic_processed,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")
        except Exception as e:
            print(f"Error initializing TableEvaluator: {str(e)}")
            print("Data types in real data:", self.real_processed.dtypes)
            print("Data types in synthetic data:", self.synthetic_processed.dtypes)
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
                # Combine unique values from both datasets
                unique_values = pd.concat([real_processed[col], synthetic_processed[col]]).unique()

                # Create and fit label encoder
                le = LabelEncoder()
                le.fit(unique_values)

                # Transform both datasets
                real_processed[col] = le.transform(real_processed[col])
                synthetic_processed[col] = le.transform(synthetic_processed[col])

                label_encoders[col] = le

        # Handle numeric columns
        numeric_cols = [col for col in real_processed.columns if col not in self.cat_cols]
        for col in numeric_cols:
            # Convert to float and handle missing values
            real_processed[col] = pd.to_numeric(real_processed[col], errors='coerce').fillna(0)
            synthetic_processed[col] = pd.to_numeric(synthetic_processed[col], errors='coerce').fillna(0)

        print("\nProcessed data types:")
        print("Real data:\n", real_processed.dtypes)
        print("\nSynthetic data:\n", synthetic_processed.dtypes)

        return real_processed, synthetic_processed

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        try:
            print("\nDEBUG - Running table-evaluator evaluation")
            print(f"Target column: {target_col}")

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