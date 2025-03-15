"""
Evaluation module for synthetic data quality assessment
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

class DataEvaluator:
    """Evaluates quality of synthetic data compared to real data"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Initialize with real and synthetic datasets"""
        print("\nDEBUG - DataEvaluator initialization:")
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

        # Initialize table-evaluator adapter lazily to avoid circular imports
        self._table_evaluator = None

        print(f"Final evaluation columns: {self.real_data.columns.tolist()}")

    def _get_evaluator(self):
        """Lazy initialization of table evaluator adapter"""
        if self._table_evaluator is None:
            from .table_evaluator_adapter import TableEvaluatorAdapter
            self._table_evaluator = TableEvaluatorAdapter(
                real_data=self.real_data,
                synthetic_data=self.synthetic_data
            )
        return self._table_evaluator

    def evaluate(self, target_col: str) -> dict:
        """Run comprehensive evaluation using table-evaluator"""
        return self._get_evaluator().evaluate_all(target_col)

    def generate_evaluation_plots(self):
        """Generate visual evaluation plots using table-evaluator"""
        return self._get_evaluator().get_visual_evaluation()

    def calculate_jsd(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calculate Jensen-Shannon Divergence between real and synthetic data distributions"""
        real_hist, bins = np.histogram(real_col, bins=50, density=True)
        synth_hist, _ = np.histogram(synthetic_col, bins=bins, density=True)

        real_hist = np.clip(real_hist, 1e-10, None)
        synth_hist = np.clip(synth_hist, 1e-10, None)
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()

        return jensenshannon(real_hist, synth_hist)

    def calculate_wasserstein(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calculate Wasserstein Distance between real and synthetic data distributions"""
        return wasserstein_distance(real_col.values, synthetic_col.values)

    def calculate_distribution_divergence(self) -> dict:
        """Calculate distribution divergence metrics"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        divergence_metrics = {}
        for col in numerical_cols:
            jsd = self.calculate_jsd(self.real_data[col], self.synthetic_data[col])
            wd = self.calculate_wasserstein(self.real_data[col], self.synthetic_data[col])
            divergence_metrics[f'{col}_jsd'] = jsd
            divergence_metrics[f'{col}_wasserstein'] = wd
        return divergence_metrics