import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns

class DataEvaluator:
    """Evaluates quality of synthetic data compared to real data"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Initialize with real and synthetic datasets"""
        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Ensure all columns from real data exist in synthetic data
        missing_cols = set(real_data.columns) - set(synthetic_data.columns)
        if missing_cols:
            raise ValueError(f"Synthetic data is missing columns: {missing_cols}")

        # Ensure column order matches between real and synthetic data
        self.synthetic_data = self.synthetic_data[self.real_data.columns]

    def preprocess_features(self, X: pd.DataFrame, is_training: bool = True, scaler=None, encoders=None):
        """Preprocess features while preserving column names"""
        X = X.copy()
        result_df = pd.DataFrame(index=X.index)

        if is_training:
            scaler = StandardScaler()
            encoders = {}

        # Process each column individually to maintain feature names
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Handle numerical columns
                if is_training:
                    result_df[col] = scaler.fit_transform(X[[col]])
                else:
                    result_df[col] = scaler.transform(X[[col]])
            else:
                # Handle categorical columns
                if is_training:
                    encoder = LabelEncoder()
                    # Get unique values and add 'Other'
                    unique_values = list(X[col].unique()) + ['Other']
                    encoder.fit(unique_values)
                    encoders[col] = encoder

                # Map unseen categories to 'Other'
                X[col] = X[col].map(lambda x: 'Other' if x not in encoders[col].classes_ else x)
                result_df[col] = encoders[col].transform(X[col])

        if is_training:
            return result_df, scaler, encoders
        return result_df

    def evaluate_ml_utility(self, target_column: str, task_type: str = 'classification', test_size: float = 0.2) -> dict:
        """Evaluate ML utility using Train-Synthetic-Test-Real (TSTR) methodology"""
        try:
            # Prepare features and target
            X_real = self.real_data.drop(columns=[target_column])
            y_real = self.real_data[target_column]
            X_synthetic = self.synthetic_data.drop(columns=[target_column])
            y_synthetic = self.synthetic_data[target_column]

            # Split real data
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=test_size, random_state=42
            )

            # Preprocess features while maintaining column names
            X_train_real_processed, scaler, encoders = self.preprocess_features(X_train_real, is_training=True)
            X_test_real_processed = self.preprocess_features(X_test_real, is_training=False, scaler=scaler, encoders=encoders)
            X_synthetic_processed = self.preprocess_features(X_synthetic, is_training=False, scaler=scaler, encoders=encoders)

            # Handle target variable
            if task_type == 'classification':
                target_encoder = LabelEncoder()
                # Include all possible categories including 'Other'
                unique_values = list(set(y_real.unique()) | set(y_synthetic.unique()) | {'Other'})
                target_encoder.fit(unique_values)

                y_train_real = pd.Series(y_train_real).map(lambda x: 'Other' if x not in unique_values else x)
                y_test_real = pd.Series(y_test_real).map(lambda x: 'Other' if x not in unique_values else x)
                y_synthetic = pd.Series(y_synthetic).map(lambda x: 'Other' if x not in unique_values else x)

                y_train_real = target_encoder.transform(y_train_real)
                y_test_real = target_encoder.transform(y_test_real)
                y_synthetic = target_encoder.transform(y_synthetic)

            # Initialize models
            if task_type == 'classification':
                real_model = RandomForestClassifier(n_estimators=100, random_state=42)
                synthetic_model = RandomForestClassifier(n_estimators=100, random_state=42)
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:
                real_model = RandomForestRegressor(n_estimators=100, random_state=42)
                synthetic_model = RandomForestRegressor(n_estimators=100, random_state=42)
                metric_func = r2_score
                metric_name = 'r2_score'

            # Train and evaluate
            real_model.fit(X_train_real_processed, y_train_real)
            real_pred = real_model.predict(X_test_real_processed)
            real_score = metric_func(y_test_real, real_pred)

            synthetic_model.fit(X_synthetic_processed, y_synthetic)
            synthetic_pred = synthetic_model.predict(X_test_real_processed)
            synthetic_score = metric_func(y_test_real, synthetic_pred)

            relative_performance = (synthetic_score / real_score) * 100 if real_score != 0 else 0

            return {
                f'real_model_{metric_name}': real_score,
                f'synthetic_model_{metric_name}': synthetic_score,
                'relative_performance_percentage': relative_performance
            }

        except Exception as e:
            print(f"Error in ML utility evaluation: {str(e)}")
            raise

    def statistical_similarity(self) -> dict:
        """Calculate statistical similarity metrics"""
        metrics = {}
        numerical_cols = [col for col in self.real_data.columns 
                         if pd.api.types.is_numeric_dtype(self.real_data[col])]

        for col in numerical_cols:
            statistic, pvalue = stats.ks_2samp(
                self.real_data[col],
                self.synthetic_data[col]
            )
            metrics[f'ks_statistic_{col}'] = statistic
            metrics[f'ks_pvalue_{col}'] = pvalue

        return metrics

    def correlation_similarity(self) -> float:
        """Compare correlation matrices of real and synthetic data"""
        numerical_cols = [col for col in self.real_data.columns 
                         if pd.api.types.is_numeric_dtype(self.real_data[col])]

        if not numerical_cols:
            return 0.0

        real_corr = self.real_data[numerical_cols].corr()
        synth_corr = self.synthetic_data[numerical_cols].corr()
        correlation_distance = np.linalg.norm(real_corr - synth_corr)
        max_possible_distance = np.sqrt(2 * len(numerical_cols))
        correlation_similarity = 1 - (correlation_distance / max_possible_distance)

        return correlation_similarity

    def column_statistics(self) -> pd.DataFrame:
        """Compare basic statistics for each column"""
        numerical_cols = [col for col in self.real_data.columns 
                         if pd.api.types.is_numeric_dtype(self.real_data[col])]

        if not numerical_cols:
            return pd.DataFrame()

        stats_real = self.real_data[numerical_cols].describe()
        stats_synthetic = self.synthetic_data[numerical_cols].describe()

        comparison = pd.DataFrame({
            'real_mean': stats_real.loc['mean'],
            'synthetic_mean': stats_synthetic.loc['mean'],
            'real_std': stats_real.loc['std'],
            'synthetic_std': stats_synthetic.loc['std'],
            'real_min': stats_real.loc['min'],
            'synthetic_min': stats_synthetic.loc['min'],
            'real_max': stats_real.loc['max'],
            'synthetic_max': stats_synthetic.loc['max']
        })

        comparison['mean_diff_pct'] = np.abs(
            (comparison['real_mean'] - comparison['synthetic_mean']) / comparison['real_mean']
        ) * 100

        comparison['std_diff_pct'] = np.abs(
            (comparison['real_std'] - comparison['synthetic_std']) / comparison['real_std']
        ) * 100

        return comparison

    def plot_distributions(self, save_path: str = None):
        """Plot distribution comparisons for numerical columns"""
        numerical_cols = [col for col in self.real_data.columns 
                         if pd.api.types.is_numeric_dtype(self.real_data[col])]

        if not numerical_cols:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.text(0.5, 0.5, 'No numerical columns to compare', 
                   horizontalalignment='center', verticalalignment='center')
            if save_path:
                plt.savefig(save_path)
                plt.close()
            return fig

        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 5*n_cols))
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, numerical_cols):
            sns.kdeplot(data=self.real_data[col], label='Real', ax=ax)
            sns.kdeplot(data=self.synthetic_data[col], label='Synthetic', ax=ax)
            ax.set_title(f'Distribution Comparison - {col}')
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            return fig

    def evaluate_all(self, target_column: str = None, task_type: str = 'classification') -> dict:
        """Run all evaluations and return comprehensive metrics"""
        evaluation = {
            'statistical_tests': self.statistical_similarity(),
            'correlation_similarity': self.correlation_similarity(),
            'column_statistics': self.column_statistics().to_dict(),
        }

        if target_column:
            evaluation['ml_utility'] = self.evaluate_ml_utility(
                target_column=target_column,
                task_type=task_type
            )

        return evaluation