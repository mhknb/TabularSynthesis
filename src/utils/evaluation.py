import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, f_oneway
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
        print("\nDEBUG - DataEvaluator initialization:")
        print(f"Real data columns: {real_data.columns.tolist()}")
        print(f"Synthetic data columns: {synthetic_data.columns.tolist()}")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")

        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()

        # Find common columns for evaluation
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        print(f"Common columns for evaluation: {common_cols}")

        # Handle the case when no common columns exist
        if not common_cols:
            print("WARNING: No common columns found between real and synthetic data!")
            # Add dummy column to allow initialization but prevent meaningful evaluation
            self.real_data['_dummy'] = 0
            self.synthetic_data['_dummy'] = 0
            common_cols = ['_dummy']

        # Only use columns that exist in both datasets
        self.real_data = self.real_data[common_cols]
        self.synthetic_data = self.synthetic_data[common_cols]

        # Fill missing values to prevent NoneType comparison errors
        self.real_data = self.real_data.fillna(0)
        self.synthetic_data = self.synthetic_data.fillna(0)

        print(f"Final evaluation columns: {self.real_data.columns.tolist()}")
        print(f"Final shapes - Real: {self.real_data.shape}, Synthetic: {self.synthetic_data.shape}")

    def calculate_jsd(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calculate Jensen-Shannon Divergence between real and synthetic data distributions"""
        # Convert to probability distributions
        real_hist, bins = np.histogram(real_col, bins=50, density=True)
        synth_hist, _ = np.histogram(synthetic_col, bins=bins, density=True)

        # Ensure distributions sum to 1 and handle zero probabilities
        real_hist = np.clip(real_hist, 1e-10, None)
        synth_hist = np.clip(synth_hist, 1e-10, None)
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()

        return jensenshannon(real_hist, synth_hist)

    def calculate_wasserstein(self, real_col: pd.Series, synthetic_col: pd.Series) -> float:
        """Calculate Wasserstein Distance between real and synthetic data distributions"""
        return wasserstein_distance(real_col.values, synthetic_col.values)

    def plot_cumulative_distributions(self, save_path: str = None):
        """Plot cumulative distribution comparisons for numerical columns"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        n_cols = len(numerical_cols)

        if n_cols == 0:
            return None

        # Create subplots grid
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, col in enumerate(numerical_cols):
            row = idx // 3
            col_idx = idx % 3

            # Sort values for cumulative distribution
            real_sorted = np.sort(self.real_data[col].values)
            synth_sorted = np.sort(self.synthetic_data[col].values)

            # Calculate cumulative probabilities
            real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
            synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)

            # Normalize values to [-1, 1] range for comparison
            real_norm = 2 * (real_sorted - real_sorted.min()) / (real_sorted.max() - real_sorted.min()) - 1
            synth_norm = 2 * (synth_sorted - synth_sorted.min()) / (synth_sorted.max() - synth_sorted.min()) - 1

            # Plot
            axes[row, col_idx].plot(real_norm, real_cdf, label='Real Data', color='blue')
            axes[row, col_idx].plot(synth_norm, synth_cdf, label='Synthetic Data', color='orange')
            axes[row, col_idx].set_title(f'{col}')
            axes[row, col_idx].set_xlabel('Normalized Value')
            axes[row, col_idx].set_ylabel('Cumulative Probability')
            axes[row, col_idx].legend()
            axes[row, col_idx].grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_cols, n_rows * 3):
            row = idx // 3
            col_idx = idx % 3
            axes[row, col_idx].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        return fig

    def preprocess_features(self, X: pd.DataFrame, scalers=None, encoders=None):
        """Preprocess features while preserving column names"""
        print("\nDEBUG - Feature preprocessing:")
        print(f"Input data shape: {X.shape}")
        print(f"Input columns: {X.columns.tolist()}")

        # Create a copy to avoid modifying original data
        X = X.copy()
        result_df = pd.DataFrame(index=X.index)

        # Initialize transformers if not provided
        if scalers is None:
            print("Creating new scalers")
            scalers = {}
            is_training = True
        else:
            print("Using existing scalers")
            is_training = False

        if encoders is None:
            print("Creating new encoders")
            encoders = {}

        # Process numerical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            print(f"Processing numerical column: {col}")
            if col not in scalers and is_training:
                scalers[col] = StandardScaler()

            if is_training:
                result_df[col] = scalers[col].fit_transform(X[[col]])
            else:
                result_df[col] = scalers[col].transform(X[[col]])

        # Process categorical columns
        categorical_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
        for col in categorical_cols:
            print(f"Processing categorical column: {col}")
            if col not in encoders:
                print(f"Creating new encoder for {col}")
                encoder = LabelEncoder()
                unique_values = list(X[col].unique()) + ['Other']
                encoder.fit(unique_values)
                encoders[col] = encoder

            # Map unseen categories to 'Other'
            X[col] = X[col].map(lambda x: 'Other' if x not in encoders[col].classes_ else x)
            result_df[col] = encoders[col].transform(X[col])

        print(f"Output data shape: {result_df.shape}")
        print(f"Output columns: {result_df.columns.tolist()}")

        return result_df, scalers, encoders

    def evaluate_ml_utility(self, target_column: str, task_type: str = 'classification', test_size: float = 0.2) -> dict:
        """Evaluate ML utility using Train-Synthetic-Test-Real (TSTR) methodology"""
        try:
            print(f"\nDEBUG - ML Utility Evaluation:")
            print(f"Target column: {target_column}")
            print(f"Task type: {task_type}")

            # Prepare features and target
            feature_cols = [col for col in self.real_data.columns if col != target_column]
            X_real = self.real_data[feature_cols]
            y_real = self.real_data[target_column]
            X_synthetic = self.synthetic_data[feature_cols]
            y_synthetic = self.synthetic_data[target_column]

            print(f"Feature columns: {feature_cols}")

            # Split real data
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=test_size, random_state=42
            )

            # Preprocess features
            X_train_processed, scalers, encoders = self.preprocess_features(X_train_real)
            X_test_processed = self.preprocess_features(X_test_real, scalers=scalers, encoders=encoders)[0]
            X_synthetic_processed = self.preprocess_features(X_synthetic, scalers=scalers, encoders=encoders)[0]

            # Handle target variable
            if task_type == 'classification':
                target_encoder = LabelEncoder()
                unique_vals = list(set(y_real.unique()) | set(y_synthetic.unique()) | {'Other'})
                target_encoder.fit(unique_vals)

                y_train_real = pd.Series(y_train_real).map(lambda x: 'Other' if x not in unique_vals else x)
                y_test_real = pd.Series(y_test_real).map(lambda x: 'Other' if x not in unique_vals else x)
                y_synthetic = pd.Series(y_synthetic).map(lambda x: 'Other' if x not in unique_vals else x)

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

            # Train and evaluate models
            real_model.fit(X_train_processed, y_train_real)
            real_pred = real_model.predict(X_test_processed)
            real_score = metric_func(y_test_real, real_pred)

            synthetic_model.fit(X_synthetic_processed, y_synthetic)
            synthetic_pred = synthetic_model.predict(X_test_processed)
            synthetic_score = metric_func(y_test_real, synthetic_pred)

            relative_performance = (synthetic_score / real_score) * 100 if real_score != 0 else 0

            return {
                f'real_model_{metric_name}': real_score,
                f'synthetic_model_{metric_name}': synthetic_score,
                'relative_performance_percentage': relative_performance
            }

        except Exception as e:
            print(f"Error in ML utility evaluation: {str(e)}")
            print("Full error context:")
            import traceback
            traceback.print_exc()
            raise

    def statistical_similarity(self) -> dict:
        """Calculate statistical similarity metrics"""
        metrics = {}
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns

        for col in numerical_cols:
            # One-way ANOVA test
            try:
                f_statistic, anova_pvalue = f_oneway(
                    self.real_data[col],
                    self.synthetic_data[col]
                )
                metrics[f'anova_f_statistic_{col}'] = f_statistic
                metrics[f'anova_pvalue_{col}'] = anova_pvalue
            except:
                # Handle cases where ANOVA cannot be calculated
                metrics[f'anova_f_statistic_{col}'] = float('nan')
                metrics[f'anova_pvalue_{col}'] = float('nan')

        return metrics
        
    def anova_summary(self) -> pd.DataFrame:
        """Summarize one-way ANOVA results for each numerical column"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        results = []
        
        for col in numerical_cols:
            try:
                f_statistic, p_value = f_oneway(
                    self.real_data[col],
                    self.synthetic_data[col]
                )
                
                # Interpretation
                if p_value < 0.05:
                    interpretation = "Significant difference"
                else:
                    interpretation = "No significant difference"
                    
                results.append({
                    'Column': col,
                    'F_statistic': f_statistic,
                    'P_value': p_value,
                    'Interpretation': interpretation
                })
            except Exception as e:
                results.append({
                    'Column': col,
                    'F_statistic': float('nan'),
                    'P_value': float('nan'),
                    'Interpretation': f"Error: {str(e)}"
                })
                
        return pd.DataFrame(results)
    
    def correlation_similarity(self) -> float:
        """Compare correlation matrices of real and synthetic data"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
            return 0.0

        # Filter out columns with zero variance in either dataset
        valid_cols = []
        for col in numerical_cols:
            real_var = self.real_data[col].var()
            synth_var = self.synthetic_data[col].var()
            if real_var > 0 and synth_var > 0:
                valid_cols.append(col)

        if len(valid_cols) <= 1:
            # Not enough valid columns for correlation calculation
            return 0.0

        # Calculate correlation matrices with valid columns only
        real_corr = self.real_data[valid_cols].corr().fillna(0)
        synth_corr = self.synthetic_data[valid_cols].corr().fillna(0)

        # Calculate norm of difference
        try:
            correlation_distance = np.linalg.norm(real_corr - synth_corr)
            max_possible_distance = np.sqrt(2 * len(valid_cols))
            correlation_similarity = 1 - (correlation_distance / max_possible_distance)

            # Ensure result is in valid range
            return max(0.0, min(1.0, correlation_similarity))
        except:
            # If calculation fails for any reason, return 0
            return 0.0

    def column_statistics(self) -> pd.DataFrame:
        """Compare basic statistics for each column"""
        # Get columns present in both datasets
        common_cols = [col for col in self.real_data.columns if col in self.synthetic_data.columns]
        numerical_cols = self.real_data[common_cols].select_dtypes(include=['int64', 'float64']).columns

        if len(numerical_cols) == 0:
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

        # Add safe division to avoid division by zero
        comparison['mean_diff_pct'] = np.abs(
            (comparison['real_mean'] - comparison['synthetic_mean']) / 
            (comparison['real_mean'].replace(0, np.nan).fillna(1e-10))
        ) * 100

        comparison['std_diff_pct'] = np.abs(
            (comparison['real_std'] - comparison['synthetic_std']) / 
            (comparison['real_std'].replace(0, np.nan).fillna(1e-10))
        ) * 100

        return comparison

    def plot_distributions(self, save_path: str = None):
        """Plot distribution comparisons for numerical columns"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
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
        return fig

    def calculate_distribution_divergence(self) -> dict:
        """Calculate JSD and WD for numerical columns"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        divergence_metrics = {}
        for col in numerical_cols:
            jsd = self.calculate_jsd(self.real_data[col], self.synthetic_data[col])
            wd = self.calculate_wasserstein(self.real_data[col], self.synthetic_data[col])
            divergence_metrics[f'{col}_jsd'] = jsd
            divergence_metrics[f'{col}_wasserstein'] = wd
        return divergence_metrics

    def evaluate_all(self, target_column=None, task_type='classification'):
        """Run all evaluations and return combined results"""
        results = {}

        try:
            results['statistical_similarity'] = self.statistical_similarity()
        except Exception as e:
            results['statistical_similarity'] = f"Error: {str(e)}"

        try:
            results['correlation_similarity'] = self.correlation_similarity()
        except Exception as e:
            results['correlation_similarity'] = f"Error: {str(e)}"

        try:
            results['column_statistics'] = self.column_statistics()
        except Exception as e:
            results['column_statistics'] = f"Error: {str(e)}"

        try:
            results['divergence_metrics'] = self.calculate_distribution_divergence()
        except Exception as e:
            results['divergence_metrics'] = f"Error: {str(e)}"
            
        try:
            results['anova_summary'] = self.anova_summary()
        except Exception as e:
            results['anova_summary'] = f"Error: {str(e)}"

        if target_column:
            if target_column in self.real_data.columns and target_column in self.synthetic_data.columns:
                try:
                    results['ml_utility'] = self.evaluate_ml_utility(target_column, task_type)
                except Exception as e:
                    results['ml_utility'] = f"Error: {str(e)}"
            else:
                results['ml_utility'] = "Target column not found in both datasets"

        return results