"""
Adapter module for integrating table-evaluator functionality
"""
from typing import Dict, Any, Optional, List
import pandas as pd
from table_evaluator import TableEvaluator

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
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.cat_cols = cat_cols or self._infer_categorical_columns()
        self.evaluator = TableEvaluator(
            real_data=real_data,
            fake_data=synthetic_data,
            cat_cols=self.cat_cols
        )

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on number of unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            n_unique = self.real_data[col].nunique()
            if n_unique < 50 or self.real_data[col].dtype == 'object':
                categorical_columns.append(col)
        return categorical_columns

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """
        Run comprehensive evaluation using table-evaluator

        Args:
            target_col: Target column name for ML utility evaluation

        Returns:
            Dictionary containing evaluation results
        """
        # Get evaluation results
        ml_scores = self.evaluator.evaluate(target_col=target_col)

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

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        return self.evaluator.visual_evaluation()