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
        """Initialize with real and synthetic datasets"""
        print("\nDEBUG - Starting TableEvaluatorAdapter initialization")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")

        # Convert all data to numeric float64
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Force all data to float64
        for col in self.real_data.columns:
            try:
                print(f"\nProcessing column: {col}")
                # Convert both datasets to numeric
                self.real_data[col] = self.real_data[col].astype(str).astype('category').cat.codes.astype('float64')
                self.synthetic_data[col] = self.synthetic_data[col].astype(str).astype('category').cat.codes.astype('float64')

                print(f"Converted {col} successfully")
                print(f"Real dtype: {self.real_data[col].dtype}")
                print(f"Synthetic dtype: {self.synthetic_data[col].dtype}")

                # Verify the conversion
                if not isinstance(self.real_data[col].iloc[0], (np.floating, float)):
                    raise TypeError(f"Column {col} contains non-float data after conversion")

            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")
                raise

        # No categorical columns since everything is numeric now
        self.cat_cols = []

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run evaluation using table-evaluator"""
        try:
            print("\nStarting evaluation")
            print(f"Target column: {target_col}")
            print(f"Target column dtype (real): {self.real_data[target_col].dtype}")
            print(f"Target column dtype (synthetic): {self.synthetic_data[target_col].dtype}")

            # Initialize evaluator
            evaluator = TableEvaluator(
                real=self.real_data,
                fake=self.synthetic_data,
                cat_cols=self.cat_cols  # Empty list since all data is numeric
            )

            # Run evaluation
            print("Running evaluation...")
            results = evaluator.evaluate(target_col=target_col)
            print("Evaluation completed")

            return {
                'classifier_scores': results.get('Classifier F1-scores', None),
                'privacy': {
                    'Duplicate rows between sets (real/fake)': results.get('Duplicate rows between sets (real/fake)', (0, 0)),
                    'nearest neighbor mean': results.get('nearest neighbor mean', 0),
                    'nearest neighbor std': results.get('nearest neighbor std', 0)
                },
                'correlation': {
                    'Column Correlation Distance RMSE': results.get('Column Correlation Distance RMSE', 0),
                    'Column Correlation distance MAE': results.get('Column Correlation distance MAE', 0)
                },
                'similarity': {
                    'basic statistics': results.get('basic statistics', 0),
                    'Correlation column correlations': results.get('Correlation column correlations', 0),
                    'Mean Correlation between fake and real columns': results.get('Mean Correlation between fake and real columns', 0),
                    '1 - MAPE Estimator results': results.get('1 - MAPE Estimator results', 0),
                    '1 - MAPE 5 PCA components': results.get('1 - MAPE 5 PCA components', 0),
                    'Similarity Score': results.get('Similarity Score', 0)
                }
            }

        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")
            print("\nDebug information:")
            print("Data types:")
            print(self.real_data.dtypes)
            print("\nSample data:")
            print(self.real_data.head())
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            print("\nGenerating visual evaluation")
            evaluator = TableEvaluator(
                real=self.real_data,
                fake=self.synthetic_data,
                cat_cols=self.cat_cols
            )
            return evaluator.visual_evaluation()
        except Exception as e:
            print(f"Error in visual evaluation: {str(e)}")
            raise