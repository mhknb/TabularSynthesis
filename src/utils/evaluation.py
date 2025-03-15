import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon, cdist
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.decomposition import PCA

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

        # Find common columns for evaluation
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        print(f"Common columns for evaluation: {common_cols}")

        # Handle the case when no common columns exist
        if not common_cols:
            raise ValueError("No common columns found between real and synthetic data!")

        # Only use columns that exist in both datasets
        self.real_data = self.real_data[common_cols]
        self.synthetic_data = self.synthetic_data[common_cols]

        # Remove duplicates from synthetic data
        orig_len = len(self.synthetic_data)
        self.synthetic_data = self.synthetic_data.drop_duplicates()
        dropped_dupes = orig_len - len(self.synthetic_data)
        print(f"Removed {dropped_dupes} duplicate rows from synthetic data")

        # Fill missing values with appropriate defaults based on data type
        for col in self.real_data.columns:
            if self.real_data[col].dtype in ['int64', 'float64']:
                self.real_data[col] = self.real_data[col].fillna(0)
                self.synthetic_data[col] = self.synthetic_data[col].fillna(0)
            else:
                self.real_data[col] = self.real_data[col].fillna('Unknown')
                self.synthetic_data[col] = self.synthetic_data[col].fillna('Unknown')

        print(f"Final evaluation columns: {self.real_data.columns.tolist()}")

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

    def get_duplicate_count(self):
        """Count duplicate rows between real and synthetic data"""
        print("\nDEBUG - Calculating duplicate counts")
        real_duplicates = len(self.real_data[self.real_data.duplicated()])
        synth_duplicates = len(self.synthetic_data[self.synthetic_data.duplicated()])
        print(f"Found duplicates - Real: {real_duplicates}, Synthetic: {synth_duplicates}")
        return real_duplicates, synth_duplicates

    def preprocess_data_for_distance(self):
        """Preprocess data for distance calculations"""
        print("\nDEBUG - Preprocessing data for distance calculations")
        result_real = pd.DataFrame()
        result_synthetic = pd.DataFrame()

        # Process numerical columns
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()

        if len(numerical_cols) > 0:
            print(f"Processing numerical columns: {numerical_cols.tolist()}")
            numerical_data_real = self.real_data[numerical_cols]
            numerical_data_synthetic = self.synthetic_data[numerical_cols]

            # Fit on concatenated data to ensure same scale
            combined_numerical = pd.concat([numerical_data_real, numerical_data_synthetic])
            scaler.fit(combined_numerical)

            # Transform both datasets
            result_real[numerical_cols] = scaler.transform(numerical_data_real)
            result_synthetic[numerical_cols] = scaler.transform(numerical_data_synthetic)

        # Process categorical columns
        categorical_cols = self.real_data.select_dtypes(exclude=['int64', 'float64']).columns

        if len(categorical_cols) > 0:
            print(f"Processing categorical columns: {categorical_cols.tolist()}")
            for col in categorical_cols:
                encoder = LabelEncoder()
                combined_categories = pd.concat([self.real_data[col], self.synthetic_data[col]]).unique()
                encoder.fit(combined_categories)

                result_real[col] = encoder.transform(self.real_data[col])
                result_synthetic[col] = encoder.transform(self.synthetic_data[col])

        print(f"Preprocessed data shapes - Real: {result_real.shape}, Synthetic: {result_synthetic.shape}")
        return result_real, result_synthetic

    def nearest_neighbor_analysis(self, max_samples=20000):
        """Calculate nearest neighbor distances between real and synthetic data"""
        try:
            print("\nDEBUG - Starting nearest neighbor analysis")
            # Preprocess data
            real_processed, synth_processed = self.preprocess_data_for_distance()

            # Sample data if needed
            real_sample = real_processed.sample(n=min(len(real_processed), max_samples))
            synth_sample = synth_processed.sample(n=min(len(synth_processed), max_samples))

            print(f"Sample sizes - Real: {len(real_sample)}, Synthetic: {len(synth_sample)}")

            # Calculate distances
            distances = cdist(real_sample, synth_sample, metric='euclidean')
            min_distances = distances.min(axis=1)

            mean_dist = float(min_distances.mean())
            std_dist = float(min_distances.std())

            print(f"Distance metrics - Mean: {mean_dist:.4f}, Std: {std_dist:.4f}")
            return mean_dist, std_dist
        except Exception as e:
            print(f"Error in nearest neighbor analysis: {str(e)}")
            raise

    def calculate_correlation_distance(self):
        """Calculate correlation matrix distance metrics"""
        numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) < 2:
            return 0.0, 0.0

        real_corr = self.real_data[numerical_cols].corr().fillna(0)
        synth_corr = self.synthetic_data[numerical_cols].corr().fillna(0)

        rmse = np.sqrt(((real_corr - synth_corr) ** 2).mean().mean())
        mae = np.abs(real_corr - synth_corr).mean().mean()

        return rmse, mae


    def evaluate_ml_utility(self, target_column: str, task_type: str = 'classification', test_size: float = 0.2) -> dict:
        """Evaluate ML utility using both TSTR and combined training methodologies"""
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
                synthetic_model = RandomForestClassifier(n_estimators=100, random_state=42)
                combined_model = RandomForestClassifier(n_estimators=100, random_state=42)
                real_model = RandomForestClassifier(n_estimators=100, random_state=42)
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:
                synthetic_model = RandomForestRegressor(n_estimators=100, random_state=42)
                combined_model = RandomForestRegressor(n_estimators=100, random_state=42)
                real_model = RandomForestRegressor(n_estimators=100, random_state=42)
                metric_func = r2_score
                metric_name = 'r2_score'

            # Evaluation 1: Train on synthetic, test on real (TSTR)
            synthetic_model.fit(X_synthetic_processed, y_synthetic)
            synthetic_pred = synthetic_model.predict(X_test_processed)
            synthetic_score = metric_func(y_test_real, synthetic_pred)

            # Evaluation 2: Train on combined (synthetic + real), test on real
            # Combine training data
            X_combined = np.vstack([X_train_processed, X_synthetic_processed])
            y_combined = np.concatenate([y_train_real, y_synthetic])

            combined_model.fit(X_combined, y_combined)
            combined_pred = combined_model.predict(X_test_processed)
            combined_score = metric_func(y_test_real, combined_pred)

            # Evaluation 3: Train on real only for baseline comparison
            real_model.fit(X_train_processed, y_train_real)
            real_pred = real_model.predict(X_test_processed)
            real_score = metric_func(y_test_real, real_pred)

            # Calculate relative performances
            synthetic_relative = (synthetic_score / real_score) * 100 if real_score != 0 else 0
            combined_relative = (combined_score / real_score) * 100 if real_score != 0 else 0

            return {
                f'real_model_{metric_name}': real_score,
                f'synthetic_model_{metric_name}': synthetic_score,
                f'combined_model_{metric_name}': combined_score,
                'synthetic_relative_performance': synthetic_relative,
                'combined_relative_performance': combined_relative
            }

        except Exception as e:
            print(f"Error in ML utility evaluation: {str(e)}")
            raise

    def statistical_similarity(self) -> dict:
        """Calculate statistical similarity metrics"""
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

    def evaluate_classifiers(self, target_col: str, test_size: float = 0.2):
        """Evaluate multiple classifiers on both real and synthetic data"""
        try:
            print("\nDEBUG - Starting classifier evaluation")
            print(f"Target column: {target_col}")

            # Prepare features and target
            feature_cols = [col for col in self.real_data.columns if col != target_col]
            X_real = self.real_data[feature_cols]
            y_real = self.real_data[target_col]
            X_synthetic = self.synthetic_data[feature_cols]
            y_synthetic = self.synthetic_data[target_col]

            # Split real data
            X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
                X_real, y_real, test_size=test_size, random_state=42
            )

            # Preprocess features
            X_train_processed, scalers, encoders = self.preprocess_features(X_train_real)
            X_test_processed = self.preprocess_features(X_test_real, scalers=scalers, encoders=encoders)[0]
            X_synthetic_processed = self.preprocess_features(X_synthetic, scalers=scalers, encoders=encoders)[0]

            # Handle target variable
            target_encoder = LabelEncoder()
            unique_vals = list(set(y_real.unique()) | set(y_synthetic.unique()) | {'Other'})
            target_encoder.fit(unique_vals)

            y_train_real = pd.Series(y_train_real).map(lambda x: 'Other' if x not in unique_vals else x)
            y_test_real = pd.Series(y_test_real).map(lambda x: 'Other' if x not in unique_vals else x)
            y_synthetic = pd.Series(y_synthetic).map(lambda x: 'Other' if x not in unique_vals else x)

            y_train_real = target_encoder.transform(y_train_real)
            y_test_real = target_encoder.transform(y_test_real)
            y_synthetic = target_encoder.transform(y_synthetic)

            # Initialize classifiers
            classifiers = {
                'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),
                'RandomForestClassifier': RandomForestClassifier(random_state=42)
            }

            results = []
            for name, clf in classifiers.items():
                print(f"\nEvaluating {name}")
                try:
                    # Train and evaluate on real data
                    clf_real = clf.__class__(**clf.get_params())
                    clf_real.fit(X_train_processed, y_train_real)
                    real_pred = clf_real.predict(X_test_processed)
                    f1_real = f1_score(y_test_real, real_pred, average='weighted')

                    # Train on synthetic and evaluate on real test
                    clf_synthetic = clf.__class__(**clf.get_params())
                    clf_synthetic.fit(X_synthetic_processed, y_synthetic)
                    synthetic_pred = clf_synthetic.predict(X_test_processed)
                    f1_synthetic = f1_score(y_test_real, synthetic_pred, average='weighted')

                    # Calculate Jaccard similarity between predictions
                    jaccard = len(set(real_pred) & set(synthetic_pred)) / len(set(real_pred) | set(synthetic_pred)) if len(set(real_pred) | set(synthetic_pred)) >0 else 0

                    print(f"{name} scores - Real: {f1_real:.4f}, Synthetic: {f1_synthetic:.4f}, Jaccard: {jaccard:.4f}")

                    # Only add one row per classifier with both scores
                    results.append({
                        'index': name,
                        'f1_real': f1_real,
                        'f1_fake': f1_synthetic,
                        'jaccard_similarity': jaccard
                    })

                except Exception as e:
                    print(f"Error evaluating {name}: {str(e)}")
                    continue

            if not results:
                raise ValueError("No classifier evaluations completed successfully")

            return pd.DataFrame(results).set_index('index')

        except Exception as e:
            print(f"Error in classifier evaluation: {str(e)}")
            raise

    def evaluate_all(self, target_col: str) -> dict:
        """Run all evaluations and return combined results"""
        print("\nDEBUG - Starting comprehensive evaluation")
        results = {}

        try:
            # Classifier evaluation
            print("\nRunning classifier evaluation...")
            results['classifier_scores'] = self.evaluate_classifiers(target_col)
            print("Classifier evaluation completed successfully")
        except Exception as e:
            print(f"Error in classifier evaluation: {str(e)}")
            results['classifier_scores'] = None

        try:
            # Privacy metrics
            print("\nCalculating privacy metrics...")
            real_dupes, synth_dupes = self.get_duplicate_count()
            nn_mean, nn_std = self.nearest_neighbor_analysis()

            results['privacy'] = {
                'Duplicate rows between sets (real/fake)': (real_dupes, synth_dupes),
                'nearest neighbor mean': nn_mean,
                'nearest neighbor std': nn_std
            }
            print("Privacy metrics calculated successfully")
        except Exception as e:
            print(f"Error in privacy metrics: {str(e)}")
            results['privacy'] = None

        try:
            # Correlation metrics
            print("\nCalculating correlation metrics...")
            rmse, mae = self.calculate_correlation_distance()
            results['correlation'] = {
                'Column Correlation Distance RMSE': rmse,
                'Column Correlation distance MAE': mae
            }
            print("Correlation metrics calculated successfully")
        except Exception as e:
            print(f"Error in correlation metrics: {str(e)}")
            results['correlation'] = None

        try:
            # Additional similarity metrics
            print("\nCalculating similarity metrics...")
            results['similarity'] = self.calculate_similarity_score()
            print("Similarity metrics calculated successfully")
        except Exception as e:
            print(f"Error in similarity metrics: {str(e)}")
            results['similarity'] = None

        print("\nComprehensive evaluation completed")
        return results

    def calculate_basic_statistics(self) -> float:
        """Calculate basic statistics similarity"""
        try:
            numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) == 0:
                return 1.0  # Return perfect score if no numerical columns

            stats_real = self.real_data[numerical_cols].describe()
            stats_synthetic = self.synthetic_data[numerical_cols].describe()

            # Compare means and stds
            mean_diff = np.abs((stats_real.loc['mean'] - stats_synthetic.loc['mean']) / stats_real.loc['mean'].replace(0, np.nan).fillna(1e-10))
            std_diff = np.abs((stats_real.loc['std'] - stats_synthetic.loc['std']) / stats_real.loc['std'].replace(0, np.nan).fillna(1e-10))

            # Calculate similarity score (1 - average difference)
            mean_similarity = 1 - mean_diff.mean()
            std_similarity = 1 - std_diff.mean()

            return (mean_similarity + std_similarity) / 2
        except Exception as e:
            print(f"Error in basic statistics calculation: {str(e)}")
            return 0.0

    def calculate_column_correlations(self) -> float:
        """Calculate correlation matrix similarity"""
        try:
            numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) < 2:
                return 1.0  # Return perfect score if not enough numerical columns

            real_corr = self.real_data[numerical_cols].corr().fillna(0)
            synth_corr = self.synthetic_data[numerical_cols].corr().fillna(0)

            # Calculate correlation similarity
            correlation_distance = np.linalg.norm(real_corr - synth_corr)
            max_possible_distance = np.sqrt(2 * len(numerical_cols))  # Maximum possible Frobenius norm
            correlation_similarity = 1 - (correlation_distance / max_possible_distance)

            return max(0.0, min(1.0, correlation_similarity))
        except Exception as e:
            print(f"Error in correlation calculation: {str(e)}")
            return 0.0

    def calculate_mean_correlation(self) -> float:
        """Calculate mean correlation between real and synthetic columns"""
        try:
            numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) == 0:
                return 1.0

            correlations = []
            for col in numerical_cols:
                corr = np.corrcoef(self.real_data[col], self.synthetic_data[col])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            return np.mean(correlations) if correlations else 0.0
        except Exception as e:
            print(f"Error in mean correlation calculation: {str(e)}")
            return 0.0

    def calculate_mape_estimator(self) -> float:
        """Calculate 1 - MAPE for numerical columns"""
        try:
            numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) == 0:
                return 1.0

            mapes = []
            for col in numerical_cols:
                real_values = self.real_data[col].values
                synth_values = self.synthetic_data[col].values

                # Calculate MAPE avoiding division by zero
                denominator = np.abs(real_values)
                mask = denominator != 0
                if mask.any():
                    mape = np.mean(np.abs((real_values[mask] - synth_values[mask]) / denominator[mask]))
                    mapes.append(mape)

            return 1 - np.mean(mapes) if mapes else 0.0
        except Exception as e:
            print(f"Error in MAPE calculation: {str(e)}")
            return 0.0

    def calculate_pca_similarity(self, n_components: int = 5) -> float:
        """Calculate similarity using PCA components"""
        try:
            numerical_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) < n_components:
                return 1.0

            # Standardize the data
            real_data_std = StandardScaler().fit_transform(self.real_data[numerical_cols])
            synth_data_std = StandardScaler().fit_transform(self.synthetic_data[numerical_cols])

            # Fit PCA on real data
            pca_real = PCA(n_components=n_components)
            pca_synth = PCA(n_components=n_components)

            # Get explained variance ratios
            real_ratios = pca_real.fit(real_data_std).explained_variance_ratio_
            synth_ratios = pca_synth.fit(synth_data_std).explained_variance_ratio_

            # Calculate similarity as 1 - MAPE of explained variance ratios
            mape = np.mean(np.abs(real_ratios - synth_ratios) / real_ratios)
            return 1 - mape
        except Exception as e:
            print(f"Error in PCA similarity calculation: {str(e)}")
            return 0.0

    def calculate_similarity_score(self) -> dict:
        """Calculate overall similarity metrics"""
        results = {
            'basic statistics': self.calculate_basic_statistics(),
            'Correlation column correlations': self.calculate_column_correlations(),
            'Mean Correlation between fake and real columns': self.calculate_mean_correlation(),
            '1 - MAPE Estimator results': self.calculate_mape_estimator(),
            '1 - MAPE 5 PCA components': self.calculate_pca_similarity(),
        }

        # Calculate overall similarity score as weighted average
        weights = {
            'basic statistics': 0.3,
            'Correlation column correlations': 0.2,
            'Mean Correlation between fake and real columns': 0.2,
            '1 - MAPE Estimator results': 0.15,
            '1 - MAPE 5 PCA components': 0.15
        }

        similarity_score = sum(score * weights[metric] for metric, score in results.items())
        results['Similarity Score'] = similarity_score

        return results