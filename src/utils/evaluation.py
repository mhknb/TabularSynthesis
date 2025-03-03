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
        self.real_data = real_data
        self.synthetic_data = synthetic_data

        # Ensure all columns from real data exist in synthetic data
        missing_cols = set(real_data.columns) - set(synthetic_data.columns)
        if missing_cols:
            raise ValueError(f"Synthetic data is missing columns: {missing_cols}")

    def preprocess_features(self, X_train, X_test=None):
        """Preprocess features by handling numerical, ordinal, and categorical data"""
        # Create a copy to avoid modifying original data
        X_train = X_train.copy()
        if X_test is not None:
            X_test = X_test.copy()

        # Store column order for consistent feature ordering
        self.feature_names = X_train.columns.tolist()

        # Initialize transformers
        scaler = StandardScaler()
        encoders = {}

        # Transform features
        transformed_train = pd.DataFrame(index=X_train.index)

        for col in self.feature_names:
            if X_train[col].dtype in [np.int64, np.float64] or pd.api.types.is_numeric_dtype(X_train[col]):
                # Handle numerical and ordinal columns
                transformed_train[col] = scaler.fit_transform(X_train[[col]]).flatten()
            else:
                # Handle categorical columns
                encoder = LabelEncoder()
                # Get unique values from both train and test
                unique_values = set(X_train[col].unique())
                if X_test is not None:
                    unique_values.update(X_test[col].unique())

                # Add 'Other' category
                all_categories = list(unique_values) + ['Other']

                # Map rare categories to 'Other'
                X_train[col] = X_train[col].map(lambda x: 'Other' if x not in unique_values else x)

                # Fit encoder and transform
                encoder.fit(all_categories)
                encoders[col] = encoder
                transformed_train[col] = encoder.transform(X_train[col])

        if X_test is not None:
            transformed_test = pd.DataFrame(index=X_test.index)
            for col in self.feature_names:
                if X_test[col].dtype in [np.int64, np.float64] or pd.api.types.is_numeric_dtype(X_test[col]):
                    transformed_test[col] = scaler.transform(X_test[[col]]).flatten()
                else:
                    # Map unseen categories to 'Other'
                    X_test[col] = X_test[col].map(lambda x: 'Other' if x not in encoders[col].classes_ else x)
                    transformed_test[col] = encoders[col].transform(X_test[col])

            return transformed_train.values, transformed_test.values, scaler, encoders

        return transformed_train.values, scaler, encoders

    def evaluate_ml_utility(self, target_column: str, task_type: str = 'classification', test_size: float = 0.2) -> dict:
        """
        Evaluate ML utility using Train-Synthetic-Test-Real (TSTR) methodology
        """
        try:
            # Verify target column exists in both datasets
            if target_column not in self.synthetic_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in synthetic data")

            # Prepare features and target
            X_real = self.real_data.drop(columns=[target_column])
            y_real = self.real_data[target_column]
            X_synthetic = self.synthetic_data.drop(columns=[target_column])
            y_synthetic = self.synthetic_data[target_column]

            # Split real data into train and test sets
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=test_size, random_state=42
            )

            # Preprocess training data and get transformers
            X_train_real_processed, X_test_real_processed, scaler, encoders = self.preprocess_features(X_train_real, X_test_real)

            # Create synthetic features DataFrame with same column order
            X_synthetic_aligned = X_synthetic[self.feature_names]
            X_synthetic_processed = pd.DataFrame(index=X_synthetic.index)

            # Transform synthetic data using the same transformers
            for col in self.feature_names:
                if X_synthetic_aligned[col].dtype in [np.int64, np.float64] or pd.api.types.is_numeric_dtype(X_synthetic_aligned[col]):
                    X_synthetic_processed[col] = scaler.transform(X_synthetic_aligned[[col]]).flatten()
                else:
                    # Map unseen categories to 'Other'
                    X_synthetic_aligned[col] = X_synthetic_aligned[col].map(lambda x: 'Other' if x not in encoders[col].classes_ else x)
                    X_synthetic_processed[col] = encoders[col].transform(X_synthetic_aligned[col])

            # Handle categorical target variable
            if task_type == 'classification':
                target_encoder = LabelEncoder()
                # Get all unique values from both real and synthetic data
                unique_values = set(y_real.unique())
                unique_values.update(y_synthetic.unique())

                # Add 'Other' category to encoder classes
                all_categories = list(unique_values) + ['Other']
                target_encoder.fit(all_categories)

                # Replace rare categories with 'Other'
                y_train_real = pd.Series(y_train_real).map(lambda x: 'Other' if x not in unique_values else x)
                y_test_real = pd.Series(y_test_real).map(lambda x: 'Other' if x not in unique_values else x)
                y_synthetic = pd.Series(y_synthetic).map(lambda x: 'Other' if x not in unique_values else x)

                # Encode target variables
                y_train_real = target_encoder.transform(y_train_real)
                y_test_real = target_encoder.transform(y_test_real)
                y_synthetic = target_encoder.transform(y_synthetic)

            # Initialize models based on task type
            if task_type == 'classification':
                real_model = RandomForestClassifier(n_estimators=100, random_state=42)
                synthetic_model = RandomForestClassifier(n_estimators=100, random_state=42)
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:  # regression
                real_model = RandomForestRegressor(n_estimators=100, random_state=42)
                synthetic_model = RandomForestRegressor(n_estimators=100, random_state=42)
                metric_func = r2_score
                metric_name = 'r2_score'

            # Train and evaluate real data model
            real_model.fit(X_train_real_processed, y_train_real)
            real_pred = real_model.predict(X_test_real_processed)
            real_score = metric_func(y_test_real, real_pred)

            # Train on synthetic, test on real
            synthetic_model.fit(X_synthetic_processed.values, y_synthetic)
            synthetic_pred = synthetic_model.predict(X_test_real_processed)
            synthetic_score = metric_func(y_test_real, synthetic_pred)

            # Calculate relative performance
            relative_performance = (synthetic_score / real_score) * 100

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

        # KS test for numerical and ordinal columns
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in self.synthetic_data.columns:  # Check if column exists in synthetic data
                statistic, pvalue = stats.ks_2samp(
                    self.real_data[col],
                    self.synthetic_data[col]
                )
                metrics[f'ks_statistic_{col}'] = statistic
                metrics[f'ks_pvalue_{col}'] = pvalue

        return metrics

    def correlation_similarity(self) -> float:
        """Compare correlation matrices of real and synthetic data"""
        # Get common numerical columns between real and synthetic data
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numerical_cols if col in self.synthetic_data.columns]

        if not common_cols:
            return 0.0  # Return 0 if no common numerical columns

        real_corr = self.real_data[common_cols].corr()
        synth_corr = self.synthetic_data[common_cols].corr()

        # Calculate Frobenius norm of difference
        correlation_distance = np.linalg.norm(real_corr - synth_corr)

        # Normalize to [0,1] range where 1 means perfect correlation similarity
        max_possible_distance = np.sqrt(2 * len(common_cols))  # Maximum possible Frobenius norm
        correlation_similarity = 1 - (correlation_distance / max_possible_distance)

        return correlation_similarity

    def column_statistics(self) -> pd.DataFrame:
        """Compare basic statistics for each column"""
        # Get common numerical columns
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numerical_cols if col in self.synthetic_data.columns]

        if not common_cols:
            return pd.DataFrame()  # Return empty DataFrame if no common numerical columns

        stats_real = self.real_data[common_cols].describe()
        stats_synthetic = self.synthetic_data[common_cols].describe()

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

        # Calculate relative differences
        comparison['mean_diff_pct'] = np.abs(
            (comparison['real_mean'] - comparison['synthetic_mean']) / comparison['real_mean']
        ) * 100

        comparison['std_diff_pct'] = np.abs(
            (comparison['real_std'] - comparison['synthetic_std']) / comparison['real_std']
        ) * 100

        return comparison

    def plot_distributions(self, save_path: str = None):
        """Plot distribution comparisons for numerical columns"""
        # Get common numerical columns
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numerical_cols if col in self.synthetic_data.columns]

        if not common_cols:
            # Create empty figure if no common numerical columns
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.text(0.5, 0.5, 'No numerical columns to compare', 
                   horizontalalignment='center', verticalalignment='center')
            if save_path:
                plt.savefig(save_path)
                plt.close()
            return fig

        n_cols = len(common_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 5*n_cols))
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, common_cols):
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