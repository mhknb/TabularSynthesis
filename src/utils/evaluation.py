"""
Evaluation utilities for synthetic data generation
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from table_evaluator import TableEvaluator
import matplotlib.pyplot as plt

__all__ = ['DataEvaluator']

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

        # Only use columns that exist in both datasets
        self.real_data = self.real_data[common_cols]
        self.synthetic_data = self.synthetic_data[common_cols]

        # Identify categorical columns
        self.cat_cols = self.real_data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        print(f"Categorical columns: {self.cat_cols}")

        # Initialize table evaluator
        self.table_evaluator = TableEvaluator(
            self.real_data,  # real data as first argument
            self.synthetic_data,  # synthetic data as second argument
            cat_cols=self.cat_cols  # categorical columns as a parameter
        )

    def evaluate_all(self, target_col: str = None) -> dict:
        """Run all table evaluator metrics"""
        try:
            # Run the table evaluator
            basic_metrics = self.table_evaluator.evaluate(target_col=target_col, verbose=False, notebook=False)

            # Create comprehensive metrics dictionary
            metrics = {}

            # Add visual evaluations
            try:
                fig_correlation = self.table_evaluator.correlation_plot(plot_diff=True)
                plt.close()  # Close to prevent figure leaks
                metrics['correlation_plot'] = fig_correlation
            except Exception as e:
                print(f"Error generating correlation plot: {str(e)}")

            try:
                figs_distributions = self.table_evaluator.plot_distributions()
                metrics['plot_distributions'] = figs_distributions
                plt.close('all')  # Close all figures
            except Exception as e:
                print(f"Error generating distribution plots: {str(e)}")

            try:
                fig_pairwise = self.table_evaluator.plot_pairwise()
                plt.close()  # Close to prevent figure leaks
                metrics['plot_pairwise'] = fig_pairwise
            except Exception as e:
                print(f"Error generating pairwise plot: {str(e)}")

            # Merge the basic metrics
            metrics.update(basic_metrics)

            return metrics
        except Exception as e:
            print(f"Error in table evaluation: {str(e)}")
            return {"error": str(e)}

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

            # Initialize preprocessors
            scaler = StandardScaler()
            label_encoders = {}

            # Process numerical features
            numerical_cols = X_train_real.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 0:
                X_train_real[numerical_cols] = scaler.fit_transform(X_train_real[numerical_cols])
                X_test_real[numerical_cols] = scaler.transform(X_test_real[numerical_cols])
                X_synthetic[numerical_cols] = scaler.transform(X_synthetic[numerical_cols])

            # Process categorical features
            categorical_cols = X_train_real.select_dtypes(exclude=['int64', 'float64']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(pd.concat([X_train_real[col], X_synthetic[col]]))
                X_train_real[col] = le.transform(X_train_real[col])
                X_test_real[col] = le.transform(X_test_real[col])
                X_synthetic[col] = le.transform(X_synthetic[col])
                label_encoders[col] = le

            # Handle target variable
            if task_type == 'classification':
                target_encoder = LabelEncoder()
                target_encoder.fit(pd.concat([y_real, y_synthetic]))
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
            real_model.fit(X_train_real, y_train_real)
            real_pred = real_model.predict(X_test_real)
            real_score = metric_func(y_test_real, real_pred)

            synthetic_model.fit(X_synthetic, y_synthetic)
            synthetic_pred = synthetic_model.predict(X_test_real)
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
            return {"error": str(e)}