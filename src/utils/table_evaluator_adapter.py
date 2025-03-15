"""
Adapter module for table-evaluator with strict type enforcement and enhanced visualization
"""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from table_evaluator.plots import plot_mean_std, cdf

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

    def generate_evaluation_plots(self):
        """Generate comprehensive evaluation plots"""
        try:
            print("\nGenerating evaluation plots...")
            
            # Create figure for mean and std plots
            fig_mean_std = plt.figure(figsize=(12, 6))
            plot_mean_std(self.real_processed, self.synthetic_processed, show=False)
            plt.title("Absolute Log Mean and STDs of numeric data")
            
            # Create figure for cumulative sums
            fig_cumsums = plt.figure(figsize=(15, 10))
            nr_cols = 4
            nr_charts = len(self.real_processed.columns)
            nr_rows = max(1, nr_charts // nr_cols)
            nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows
            
            for i, col in enumerate(self.real_processed.columns):
                plt.subplot(nr_rows, nr_cols, i + 1)
                cdf(
                    self.real_processed[col],
                    self.synthetic_processed[col],
                    xlabel=col,
                    ylabel='Cumsum',
                    show=False
                )
            
            plt.tight_layout()
            return [fig_mean_std, fig_cumsums]

        except Exception as e:
            print(f"Error generating evaluation plots: {str(e)}")
            self._log_detailed_error(e)
            raise

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

            evaluation_results = {
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
                },
                'plots': self.generate_evaluation_plots()
            }

            return evaluation_results

        except Exception as e:
            print(f"\nError in evaluation: {str(e)}")
            self._log_detailed_error(e)
            raise

    [Rest of the existing methods remain unchanged...]
