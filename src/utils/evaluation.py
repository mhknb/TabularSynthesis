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

    def preprocess_features(self, X_train, X_test=None):
        """Preprocess features by handling both numerical and categorical data"""
        # Identify numerical, ordinal, and categorical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns

        # Initialize transformers
        scaler = StandardScaler()
        encoders = {}

        # Process numerical features
        X_train_num = scaler.fit_transform(X_train[numerical_cols]) if len(numerical_cols) > 0 else np.array([])

        # Process categorical features
        X_train_cat = np.zeros((len(X_train), len(categorical_cols)))
        for i, col in enumerate(categorical_cols):
            # Get unique values from both train and test sets
            unique_values = set(X_train[col].unique())
            if X_test is not None:
                unique_values.update(X_test[col].unique())

            # Handle rare categories
            all_categories = list(unique_values) + ['Other']  # Add 'Other' to all columns
            X_train[col] = X_train[col].map(lambda x: 'Other' if x not in unique_values else x)

            # Fit encoder with all possible categories including 'Other'
            encoders[col] = LabelEncoder().fit(all_categories)
            X_train_cat[:, i] = encoders[col].transform(X_train[col])

        # Combine features
        X_train_processed = np.hstack([X_train_num, X_train_cat]) if len(categorical_cols) > 0 else X_train_num

        # If test data is provided, transform it using the fitted transformers
        if X_test is not None:
            X_test_num = scaler.transform(X_test[numerical_cols]) if len(numerical_cols) > 0 else np.array([])
            X_test_cat = np.zeros((len(X_test), len(categorical_cols)))
            for i, col in enumerate(categorical_cols):
                # Map unseen categories to 'Other'
                X_test[col] = X_test[col].map(lambda x: 'Other' if x not in encoders[col].classes_ else x)
                X_test_cat[:, i] = encoders[col].transform(X_test[col])
            X_test_processed = np.hstack([X_test_num, X_test_cat]) if len(categorical_cols) > 0 else X_test_num
            return X_train_processed, X_test_processed, scaler, encoders

        return X_train_processed, scaler, encoders

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

    def evaluate_ml_utility(self, target_column: str, task_type: str = 'classification', test_size: float = 0.2) -> dict:
        """
        Evaluate ML utility using Train-Synthetic-Test-Real (TSTR) methodology
        """
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

        # Transform synthetic data using the same transformers
        X_synthetic_processed = np.zeros((len(X_synthetic), X_train_real_processed.shape[1]))

        # Process numerical features for synthetic data
        numerical_cols = X_synthetic.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            X_synthetic_num = scaler.transform(X_synthetic[numerical_cols])
            X_synthetic_processed[:, :len(numerical_cols)] = X_synthetic_num

        # Process categorical features for synthetic data
        categorical_cols = X_synthetic.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            start_idx = len(numerical_cols)
            for i, col in enumerate(categorical_cols):
                X_synthetic[col] = X_synthetic[col].map(lambda x: 'Other' if x not in encoders[col].classes_ else x)
                X_synthetic_processed[:, start_idx + i] = encoders[col].transform(X_synthetic[col])

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
        synthetic_model.fit(X_synthetic_processed, y_synthetic)
        synthetic_pred = synthetic_model.predict(X_test_real_processed)
        synthetic_score = metric_func(y_test_real, synthetic_pred)

        # Calculate relative performance
        relative_performance = (synthetic_score / real_score) * 100

        return {
            f'real_model_{metric_name}': real_score,
            f'synthetic_model_{metric_name}': synthetic_score,
            'relative_performance_percentage': relative_performance
        }

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