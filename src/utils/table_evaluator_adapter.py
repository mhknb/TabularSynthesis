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

        # Create deep copies to avoid modifying original data
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Print initial data types
        print("\nDEBUG - Initial data types:")
        for col in self.real_data.columns:
            print(f"Column '{col}':")
            print(f"  Real data type: {self.real_data[col].dtype}")
            print(f"  Sample values: {self.real_data[col].head().tolist()}")

        # Convert all columns to float64
        print("\nConverting all data to float64...")
        for col in self.real_data.columns:
            try:
                # Try direct numeric conversion first
                self.real_data[col] = pd.to_numeric(self.real_data[col], errors='coerce').astype('float64')
                self.synthetic_data[col] = pd.to_numeric(self.synthetic_data[col], errors='coerce').astype('float64')

                # Check if conversion resulted in too many NaN values
                real_nan_ratio = self.real_data[col].isna().mean()
                synthetic_nan_ratio = self.synthetic_data[col].isna().mean()

                if real_nan_ratio > 0.5 or synthetic_nan_ratio > 0.5:
                    # If too many NaNs, convert to categorical codes
                    print(f"Column {col} has too many NaNs after numeric conversion, using categorical encoding")
                    combined_values = pd.concat([self.real_data[col], self.synthetic_data[col]]).astype(str)
                    categories = pd.Categorical(combined_values).codes

                    # Split back into real and synthetic
                    n_real = len(self.real_data)
                    self.real_data[col] = categories[:n_real].astype('float64')
                    self.synthetic_data[col] = categories[n_real:].astype('float64')
            except Exception as e:
                print(f"Error in numeric conversion for {col}: {str(e)}")
                print("Converting to categorical codes...")
                # Convert to categorical codes as fallback
                combined_values = pd.concat([
                    self.real_data[col].astype(str),
                    self.synthetic_data[col].astype(str)
                ])
                categories = pd.Categorical(combined_values).codes

                # Split back into real and synthetic
                n_real = len(self.real_data)
                self.real_data[col] = categories[:n_real].astype('float64')
                self.synthetic_data[col] = categories[n_real:].astype('float64')

        # Fill any remaining NaN values with 0
        self.real_data = self.real_data.fillna(0)
        self.synthetic_data = self.synthetic_data.fillna(0)

        # Verify all data is numeric
        print("\nVerifying final data types:")
        for col in self.real_data.columns:
            print(f"Column {col}:")
            print(f"  Real final type: {self.real_data[col].dtype}")
            print(f"  Synthetic final type: {self.synthetic_data[col].dtype}")

            if not np.issubdtype(self.real_data[col].dtype, np.number):
                raise ValueError(f"Column {col} in real data is not numeric: {self.real_data[col].dtype}")
            if not np.issubdtype(self.synthetic_data[col].dtype, np.number):
                raise ValueError(f"Column {col} in synthetic data is not numeric: {self.synthetic_data[col].dtype}")

        # Consider all columns as non-categorical since we've converted everything to numeric
        self.cat_cols = []

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation using table-evaluator"""
        try:
            print(f"\nDEBUG - Starting evaluation for target column: {target_col}")
            print(f"Target column type in real data: {self.real_data[target_col].dtype}")
            print(f"Target column type in synthetic data: {self.synthetic_data[target_col].dtype}")

            # Initialize TableEvaluator
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
            print("\nData state at error:")
            print("\nReal data sample:")
            print(self.real_data.head())
            print("\nReal data info:")
            print(self.real_data.info())
            print("\nSynthetic data sample:")
            print(self.synthetic_data.head())
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