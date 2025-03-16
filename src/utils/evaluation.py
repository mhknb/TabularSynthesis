"""
Evaluation utilities for synthetic data comparison
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class DataEvaluator:
    """Evaluates quality of synthetic data compared to real data"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Initialize with real and synthetic datasets"""
        try:
            print("\nDEBUG - DataEvaluator initialization:")
            print(f"Real data shape: {real_data.shape}")
            print(f"Synthetic data shape: {synthetic_data.shape}")

            self.real_data = real_data.copy()
            self.synthetic_data = synthetic_data.copy()

            # Convert all numeric columns to float64
            numeric_columns = real_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                self.real_data[col] = self.real_data[col].astype(float)
                self.synthetic_data[col] = self.synthetic_data[col].astype(float)

            # Fill missing values
            self.real_data = self.real_data.fillna(0)
            self.synthetic_data = self.synthetic_data.fillna(0)

            print("DataEvaluator initialized successfully")
            print(f"Final data types - Real:\n{self.real_data.dtypes}")
            print(f"Final data types - Synthetic:\n{self.synthetic_data.dtypes}")

        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            raise

    def evaluate(self) -> dict:
        """Run comprehensive evaluation including plots"""
        try:
            print("\nGenerating evaluation results...")

            # Generate plots using the new optimized function
            plots = self.generate_evaluation_plots()
            if plots is None:
                return {'error': 'Plot generation failed'}

            # Get evaluation metrics
            stats_results = self.statistical_similarity()
            corr_sim = self.correlation_similarity()
            col_stats = self.column_statistics()

            return {
                'plots': plots,
                'statistics': stats_results,
                'correlation': corr_sim,
                'column_stats': col_stats
            }

        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def statistical_similarity(self) -> dict:
        """Calculate statistical similarity metrics"""
        try:
            metrics = {}
            numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns

            for col in numerical_cols:
                statistic, pvalue = stats.ks_2samp(
                    self.real_data[col],
                    self.synthetic_data[col]
                )
                metrics[f'ks_statistic_{col}'] = statistic
                metrics[f'ks_pvalue_{col}'] = pvalue

            return metrics
        except Exception as e:
            print(f"Error in statistical similarity calculation: {str(e)}")
            return {}

    def correlation_similarity(self) -> float:
        """Compare correlation matrices of real and synthetic data"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
            return 0.0

        # Filter out columns with zero variance
        valid_cols = []
        for col in numerical_cols:
            real_var = self.real_data[col].var()
            synth_var = self.synthetic_data[col].var()
            if real_var > 0 and synth_var > 0:
                valid_cols.append(col)

        if len(valid_cols) <= 1:
            return 0.0

        # Calculate correlation matrices
        real_corr = self.real_data[valid_cols].corr().fillna(0)
        synth_corr = self.synthetic_data[valid_cols].corr().fillna(0)

        try:
            correlation_distance = np.linalg.norm(real_corr - synth_corr)
            max_possible_distance = np.sqrt(2 * len(valid_cols))
            correlation_similarity = 1 - (correlation_distance / max_possible_distance)
            return max(0.0, min(1.0, correlation_similarity))
        except:
            return 0.0

    def column_statistics(self) -> pd.DataFrame:
        """Compare basic statistics for each column"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns

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

        comparison['mean_diff_pct'] = np.abs(
            (comparison['real_mean'] - comparison['synthetic_mean']) / 
            (comparison['real_mean'].replace(0, np.nan).fillna(1e-10))
        ) * 100

        return comparison

    def evaluate_ml_utility(self, target_column: str, task_type: str = 'classification', test_size: float = 0.2) -> dict:
        """Evaluate ML utility using Train-Synthetic-Test-Real (TSTR) methodology"""
        try:
            print("\nDEBUG - ML Utility Evaluation:")
            print(f"Target column: {target_column}")
            print(f"Task type: {task_type}")

            # Prepare features and target
            feature_cols = [col for col in self.real_data.columns if col != target_column]
            X_real = self.real_data[feature_cols]
            y_real = self.real_data[target_column]
            X_synthetic = self.synthetic_data[feature_cols]
            y_synthetic = self.synthetic_data[target_column]

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
                target_encoder.fit(pd.concat([y_real, y_synthetic]))
                y_train_real = target_encoder.transform(y_train_real)
                y_test_real = target_encoder.transform(y_test_real)
                y_synthetic = target_encoder.transform(y_synthetic)

            # Initialize models based on task type
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
            raise

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
            axes[row, col_idx].plot(real_norm, real_cdf, label='Real Data', color='blue', alpha=0.6)
            axes[row, col_idx].plot(synth_norm, synth_cdf, label='Synthetic Data', color='orange', alpha=0.6)
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

        if target_column:
            if target_column in self.real_data.columns and target_column in self.synthetic_data.columns:
                try:
                    results['ml_utility'] = self.evaluate_ml_utility(target_column, task_type)
                except Exception as e:
                    results['ml_utility'] = f"Error: {str(e)}"
            else:
                results['ml_utility'] = "Target column not found in both datasets"

        return results

    def plot_categorical_cdf(self, save_path: str = None):
        """Plot cumulative distribution comparisons for categorical columns, organized by related features"""
        try:
            # Get categorical columns
            categorical_cols = self.real_data.select_dtypes(include=['object', 'category']).columns
            n_cols = len(categorical_cols)

            if n_cols == 0:
                print("No categorical columns found")
                return None

            print(f"Processing {n_cols} categorical columns")

            # Group columns by feature type based on common prefixes or patterns in column names
            column_groups = {}

            # Check for different prefix patterns in column names
            for col in categorical_cols:
                # Look for common naming patterns like prefixes with underscores or other separators
                parts = col.split('_')
                if len(parts) > 1:
                    prefix = parts[0]
                    column_groups.setdefault(prefix, []).append(col)
                else:
                    # If no clear pattern, add to "other" group
                    column_groups.setdefault('other', []).append(col)

        except Exception as e:
            print(f"Error processing categorical columns: {str(e)}")
            return None

        fig_list = []

        # Process each group of related columns together
        for group_name, cols in column_groups.items():
            if not cols:
                continue

            # Determine grid layout based on number of columns in this group
            n_cols_in_group = len(cols)
            if n_cols_in_group == 1:
                fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                axes = np.array([axes])  # Convert to array for consistent indexing
            elif n_cols_in_group <= 3:
                fig, axes = plt.subplots(1, n_cols_in_group, figsize=(6*n_cols_in_group, 6))
            else:
                n_rows = (n_cols_in_group + 2) // 3  # Up to 3 columns per row
                fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
                # Make axes 2D if it isn't already
                if n_rows == 1:
                    axes = axes.reshape(1, -1)

            # Flatten the axes for easy iteration if it's a multi-dimensional array
            if n_cols_in_group > 1:
                flat_axes = axes.flatten()
            else:
                flat_axes = axes

            # Plot each column in this group
            for i, col in enumerate(cols):
                try:
                    if i < len(flat_axes):
                        ax = flat_axes[i]

                        # Get value counts and convert to proportions with error handling
                        # Get sorted values for real and synthetic data
                        real_values = self.real_data[col].sort_values().values
                        synth_values = self.synthetic_data[col].sort_values().values

                        if len(real_values) == 0 or len(synth_values) == 0:
                            print(f"Warning: Empty values for column {col}")
                            continue

                        # Calculate cumulative probabilities (like in TableEvaluator)
                        real_y = np.arange(1, len(real_values) + 1) / len(real_values)
                        synth_y = np.arange(1, len(synth_values) + 1) / len(synth_values)

                        # Plot sorted values against cumulative probabilities
                        ax.plot(real_values, real_y, linestyle='none', marker='o', 
                              label='Real', color='darkblue', markersize=8, alpha=0.6)
                        ax.plot(synth_values, synth_y, linestyle='none', marker='o', 
                              label='Fake', color='sandybrown', markersize=8, alpha=0.6)

                        # Set the plot limits and grid
                        ax.set_ylim(0, 1.05)
                        ax.grid(True, linestyle='--', alpha=0.7)

                        # Customize plot
                        ax.set_title(col)
                        ax.set_xlabel('')  # Categories shown below
                        ax.set_ylabel('Cumsum')

                        # Set x-ticks based on unique values
                        unique_values = sorted(self.real_data[col].unique())

                        if len(unique_values) <= 10:
                            ax.set_xticks(unique_values)
                            ax.set_xticklabels(unique_values, rotation=45, ha='right', fontsize=8)
                        else:
                            # Show subset of values
                            step = max(1, len(unique_values) // 5)
                            indices = list(range(0, len(unique_values), step))
                            selected_values = [unique_values[i] for i in indices]
                            ax.set_xticks(selected_values)
                            ax.set_xticklabels(selected_values, rotation=45, ha='right', fontsize=8)

                        # Add a legend but only to the first plot to avoid repetition
                        if i == 0:
                            ax.legend(loc='upper left')

                except Exception as e:
                    print(f"Error plotting column {col}: {str(e)}")
                    continue

            # Hide any unused subplots in the grid
            if n_cols_in_group > 1:
                for j in range(n_cols_in_group, len(flat_axes)):
                    flat_axes[j].set_visible(False)

            # Add an overall title for this group of columns
            fig.suptitle(f"{group_name}", fontsize=16)
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)  # Make room for the suptitle

            fig_list.append(fig)

            if save_path:
                fig.savefig(f"{save_path}_{group_name}.png")
                plt.close(fig)

        return fig_list if fig_list else None

    def generate_evaluation_plots(self):
        """Generate evaluation plots with enhanced error handling"""
        try:
            print("\nGenerating evaluation plots...")
            figures = []

            # Get numeric and categorical columns
            numeric_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = self.real_data.select_dtypes(include=['object', 'category']).columns

            n_cols = len(categorical_cols)
            if n_cols == 0:
                print("No categorical columns found")
                return None

            print(f"Processing {n_cols} categorical columns")

            # Group columns by feature type based on common prefixes or patterns in column names
            column_groups = {}

            try:
                # Check for different prefix patterns in column names
                for col in categorical_cols:
                    # Look for common naming patterns like prefixes with underscores or other separators
                    parts = col.split('_')
                    if len(parts) > 1:
                        prefix = parts[0]
                        column_groups.setdefault(prefix, []).append(col)
                    else:
                        # If no clear pattern, add to "other" group
                        column_groups.setdefault('other', []).append(col)

                fig_list = []

                # Process each group of related columns together
                for group_name, cols in column_groups.items():
                    n_cols_in_group = len(cols)
                    if n_cols_in_group > 1:
                        # Determine grid layout based on number of columns in this group
                        if n_cols_in_group <=3:
                            fig, axes = plt.subplots(1, n_cols_in_group, figsize=(6*n_cols_in_group, 6))
                        else:
                            n_rows = (n_cols_in_group + 2) // 3  # Up to 3 columns per row
                            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
                            if n_rows == 1:
                                axes = axes.reshape(1, -1)
                        flat_axes = axes.flatten()
                        for i, col in enumerate(cols):
                            try:
                                ax = flat_axes[i]
                                # Get sorted values for real and synthetic data
                                real_values = self.real_data[col].sort_values().values
                                synth_values = self.synthetic_data[col].sort_values().values

                                if len(real_values) == 0 or len(synth_values) == 0:
                                    print(f"Warning: Empty values for column {col}")
                                    continue

                                # Calculate cumulative probabilities (like in TableEvaluator)
                                real_y = np.arange(1, len(real_values) + 1) / len(real_values)
                                synth_y = np.arange(1, len(synth_values) + 1) / len(synth_values)

                                # Plot sorted values against cumulative probabilities
                                ax.plot(real_values, real_y, linestyle='none', marker='o', 
                                     label='Real', color='darkblue', markersize=8, alpha=0.6)
                                ax.plot(synth_values, synth_y, linestyle='none', marker='o', 
                                     label='Fake', color='sandybrown', markersize=8, alpha=0.6)

                                ax.set_ylim(0, 1.05)
                                ax.grid(True, linestyle='--', alpha=0.7)
                                ax.set_title(col)
                                ax.set_xlabel('')
                                ax.set_ylabel('Cumsum')

                                # Set x-ticks based on unique values
                                unique_values = sorted(self.real_data[col].unique())

                                if len(unique_values) <= 10:
                                    ax.set_xticks(unique_values)
                                    ax.set_xticklabels(unique_values, rotation=45, ha='right', fontsize=8)
                                else:
                                    # Show subset of values
                                    step = max(1, len(unique_values) // 5)
                                    indices = list(range(0, len(unique_values), step))
                                    selected_values = [unique_values[i] for i in indices]
                                    ax.set_xticks(selected_values)
                                    ax.set_xticklabels(selected_values, rotation=45, ha='right', fontsize=8)
                                if i == 0:
                                    ax.legend(loc='upper left')
                            except Exception as e:
                                print(f"Error plotting column {col}: {str(e)}")
                                continue

                        for j in range(n_cols_in_group, len(flat_axes)):
                            flat_axes[j].set_visible(False)

                        fig.suptitle(f"{group_name}", fontsize=16)
                        plt.tight_layout()
                        fig.subplots_adjust(top=0.9)
                        fig_list.append(fig)
                        if save_path:
                            fig.savefig(f"{save_path}_{group_name}.png")
                            plt.close(fig)
                    elif n_cols_in_group == 1:
                        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                        axes = np.array([axes])
                        for i, col in enumerate(cols):
                            try:
                                ax = axes[0]
                                # Get sorted values for real and synthetic data
                                real_values = self.real_data[col].sort_values().values
                                synth_values = self.synthetic_data[col].sort_values().values

                                if len(real_values) == 0 or len(synth_values) == 0:
                                    print(f"Warning: Empty values for column {col}")
                                    continue

                                # Calculate cumulative probabilities (like in TableEvaluator)
                                real_y = np.arange(1, len(real_values) + 1) / len(real_values)
                                synth_y = np.arange(1, len(synth_values) + 1) / len(synth_values)

                                # Plot sorted values against cumulative probabilities
                                ax.plot(real_values, real_y, linestyle='none', marker='o', 
                                     label='Real', color='darkblue', markersize=8, alpha=0.6)
                                ax.plot(synth_values, synth_y, linestyle='none', marker='o', 
                                     label='Fake', color='sandybrown', markersize=8, alpha=0.6)

                                ax.set_ylim(0, 1.05)
                                ax.grid(True, linestyle='--', alpha=0.7)
                                ax.set_title(col)
                                ax.set_xlabel('')
                                ax.set_ylabel('Cumsum')

                                # Set x-ticks based on unique values
                                unique_values = sorted(self.real_data[col].unique())

                                if len(unique_values) <= 10:
                                    ax.set_xticks(unique_values)
                                    ax.set_xticklabels(unique_values, rotation=45, ha='right', fontsize=8)
                                else:
                                    # Show subset of values
                                    step = max(1, len(unique_values) // 5)
                                    indices = list(range(0, len(unique_values), step))
                                    selected_values = [unique_values[i] for i in indices]
                                    ax.set_xticks(selected_values)
                                    ax.set_xticklabels(selected_values, rotation=45, ha='right', fontsize=8)
                                ax.legend(loc='upper left')
                            except Exception as e:
                                print(f"Error plotting column {col}: {str(e)}")
                                continue
                        fig_list.append(fig)
                        if save_path:
                            fig.savefig(f"{save_path}_{group_name}.png")
                            plt.close(fig)
                figures.extend(fig_list)
            except Exception as e:
                print(f"Error processing categorical columns: {str(e)}")

            # Get numeric columns
            numeric_cols = self.real_data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                # Create figure for mean and std plots
                fig_mean_std = plt.figure(figsize=(12, 6))
                ax = fig_mean_std.add_subplot(111)

                # Calculate statistics safely
                real_means = np.log(np.abs(self.real_data[numeric_cols].mean() + 1e-10))
                synth_means = np.log(np.abs(self.synthetic_data[numeric_cols].mean() + 1e-10))
                real_stds = np.log(self.real_data[numeric_cols].std() + 1e-10)
                synth_stds = np.log(self.synthetic_data[numeric_cols].std() + 1e-10)

                x = range(len(numeric_cols))
                width = 0.35

                # Plot bars
                ax.bar([i - width/2 for i in x], real_means, width, label='Real Mean', color='blue', alpha=0.5)
                ax.bar([i + width/2 for i in x], synth_means, width, label='Synthetic Mean', color='red', alpha=0.5)
                ax.bar([i - width/2 for i in x], real_stds, width, bottom=real_means, label='Real Std', color='blue', alpha=0.3)
                ax.bar([i + width/2 for i in x], synth_stds, width, bottom=synth_means, label='Synthetic Std', color='red', alpha=0.3)

                ax.set_xticks(x)
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax.set_title('Absolute Log Mean and STDs of numeric data')
                ax.legend()
                plt.tight_layout()
                figures.append(fig_mean_std)

                # Create figure for cumulative sums of numerical data
                fig_num_cumsums = plt.figure(figsize=(15, 10))
                cols_per_row = min(4, len(numeric_cols))
                rows = (len(numeric_cols) - 1) // cols_per_row + 1

                for i, col in enumerate(numeric_cols):
                    ax = fig_num_cumsums.add_subplot(rows, cols_per_row, i + 1)

                    # Calculate CDFs efficiently
                    real_col = np.sort(self.real_data[col].values)
                    synth_col = np.sort(self.synthetic_data[col].values)

                    # Use fewer points for smoother plotting
                    n_points = min(1000, len(real_col))
                    indices = np.linspace(0, len(real_col)-1, n_points).astype(int)

                    real_cdf = np.arange(1, len(real_col) + 1) / len(real_col)
                    synth_cdf = np.arange(1, len(synth_col) + 1) / len(synth_col)

                    # Plot CDFs using subset of points with bolder lines
                    ax.plot(real_col[indices], real_cdf[indices], label='Real', color='blue', alpha=0.6, linewidth=4)
                    ax.plot(synth_col[indices], synth_cdf[indices], label='Synthetic', color='red', alpha=0.6, linewidth=4)
                    ax.set_title(col)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Cumulative Probability')
                    ax.legend()

                plt.tight_layout()
                figures.append(fig_num_cumsums)

            # Generate categorical CDF plots
            categorical_cdfs = self.plot_categorical_cdf()
            if categorical_cdfs:
                figures.extend(categorical_cdfs)

            print("Successfully generated evaluation plots")
            return figures

        except Exception as e:
            print(f"Error generating evaluation plots: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            return None

    def calculate_numerical_stats(self) -> dict:
        """Calculate numerical statistics for real and synthetic data"""
        try:
            numeric_cols = self.real_data.select_dtypes(include=['float64', 'int64']).columns
            real_stats = self.real_data[numeric_cols].describe().T
            synth_stats = self.synthetic_data[numeric_cols].describe().T
            combined_stats = pd.concat([real_stats, synth_stats], axis=1, keys=['Real', 'Synthetic'])
            return combined_stats.to_dict('index')
        except Exception as e:
            print(f"Error calculating numerical stats: {str(e)}")
            return {}

    def evaluate_all(self, target_column=None, task_type='classification'):
        """Run all evaluations and return combined results"""
        try:
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

            if target_column:
                if target_column in self.real_data.columns and target_column in self.synthetic_data.columns:
                    try:
                        results['ml_utility'] = self.evaluate_ml_utility(target_column, task_type)
                    except Exception as e:
                        results['ml_utility'] = f"Error: {str(e)}"
                else:
                    results['ml_utility'] = "Target column not found in both datasets"

            try:
                results['numerical_stats'] = self.calculate_numerical_stats()
            except Exception as e:
                results['numerical_stats'] = f"Error: {str(e)}"

            return results
        except Exception as e:
            print(f"Error in evaluate_all: {str(e)}")
            return {'error': str(e)}