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
        print(f"Synthetic data shape: {synthetic_data.shape}")

        # Find common columns for evaluation
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        print(f"Common columns for evaluation: {common_cols}")

        # Only use columns that exist in both datasets
        self.real_data = real_data[common_cols].copy()
        self.synthetic_data = synthetic_data[common_cols].copy()

        # Identify categorical and numerical columns
        self.cat_cols = self.real_data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.num_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        print(f"Categorical columns: {self.cat_cols}")
        print(f"Numerical columns: {self.num_cols}")

        # Convert numerical columns to float and handle NaN values
        for col in self.num_cols:
            print(f"\nProcessing numerical column: {col}")
            print(f"Original real data type: {self.real_data[col].dtype}")
            print(f"Original synthetic data type: {self.synthetic_data[col].dtype}")

            self.real_data[col] = pd.to_numeric(self.real_data[col], errors='coerce').astype('float64')
            self.synthetic_data[col] = pd.to_numeric(self.synthetic_data[col], errors='coerce').astype('float64')

            # Fill NaN values with mean for numerical columns
            real_mean = self.real_data[col].mean()
            self.real_data[col] = self.real_data[col].fillna(real_mean)
            self.synthetic_data[col] = self.synthetic_data[col].fillna(real_mean)

            print(f"Final real data type: {self.real_data[col].dtype}")
            print(f"Final synthetic data type: {self.synthetic_data[col].dtype}")
            print(f"Sample values real: {self.real_data[col].head()}")
            print(f"Sample values synthetic: {self.synthetic_data[col].head()}")

        # Fill NaN values in categorical columns with mode
        for col in self.cat_cols:
            print(f"\nProcessing categorical column: {col}")
            print(f"Original real data type: {self.real_data[col].dtype}")
            print(f"Original synthetic data type: {self.synthetic_data[col].dtype}")

            mode_val = self.real_data[col].mode()[0]
            self.real_data[col] = self.real_data[col].fillna(mode_val)
            self.synthetic_data[col] = self.synthetic_data[col].fillna(mode_val)
            # Ensure categorical columns are string type
            self.real_data[col] = self.real_data[col].astype(str)
            self.synthetic_data[col] = self.synthetic_data[col].astype(str)

            print(f"Final real data type: {self.real_data[col].dtype}")
            print(f"Final synthetic data type: {self.synthetic_data[col].dtype}")
            print(f"Sample values real: {self.real_data[col].head()}")
            print(f"Sample values synthetic: {self.synthetic_data[col].head()}")

        print("\nDEBUG - After preprocessing:")
        print("Real data types:")
        print(self.real_data.dtypes)
        print("\nSynthetic data types:")
        print(self.synthetic_data.dtypes)

        # Initialize table evaluator
        try:
            print("\nInitializing TableEvaluator with:")
            print(f"Real data shape: {self.real_data.shape}")
            print(f"Synthetic data shape: {self.synthetic_data.shape}")
            print(f"Categorical columns: {self.cat_cols}")

            self.table_evaluator = TableEvaluator(
                self.real_data, 
                self.synthetic_data,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")
        except Exception as e:
            print(f"Error initializing TableEvaluator: {str(e)}")
            raise

    def evaluate_all(self, target_col: str = None) -> dict:
        """Run all table evaluator metrics"""
        try:
            print(f"\nDEBUG - Starting evaluation with target_col: {target_col}")

            # Run the table evaluator
            try:
                print("Running basic evaluation...")
                basic_metrics = self.table_evaluator.evaluate(target_col=target_col, verbose=False, notebook=False)
                print("Basic evaluation completed")
            except Exception as e:
                print(f"Error in basic evaluation: {str(e)}")
                return {"error": str(e)}

            # Create comprehensive metrics dictionary
            metrics = {}

            # Add visual evaluations
            try:
                print("Generating correlation plot...")
                fig_correlation = self.table_evaluator.correlation_plot(plot_diff=True)
                plt.close()  # Close to prevent figure leaks
                metrics['correlation_plot'] = fig_correlation
            except Exception as e:
                print(f"Error generating correlation plot: {str(e)}")

            try:
                print("Generating distribution plots...")
                figs_distributions = self.table_evaluator.plot_distributions()
                metrics['plot_distributions'] = figs_distributions
                plt.close('all')  # Close all figures
            except Exception as e:
                print(f"Error generating distribution plots: {str(e)}")

            try:
                print("Generating pairwise plot...")
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
            print("Full error context:")
            import traceback
            traceback.print_exc()
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
            print(f"X_real shape: {X_real.shape}, y_real shape: {y_real.shape}")
            print(f"X_synthetic shape: {X_synthetic.shape}, y_synthetic shape: {y_synthetic.shape}")
            print(f"X_real data types: {X_real.dtypes}")
            print(f"X_synthetic data types: {X_synthetic.dtypes}")
            print(f"y_real data type: {y_real.dtype}")
            print(f"y_synthetic data type: {y_synthetic.dtype}")


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
                try:
                    target_encoder = LabelEncoder()
                    target_encoder.fit(pd.concat([y_real, y_synthetic]))
                    y_train_real = target_encoder.transform(y_train_real)
                    y_test_real = target_encoder.transform(y_test_real)
                    y_synthetic = target_encoder.transform(y_synthetic)
                except Exception as e:
                    print(f"Error encoding target variable: {str(e)}")
                    return {"error": str(e)}


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
            try:
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
                print(f"Error during model training or prediction: {str(e)}")
                return {"error": str(e)}

        except Exception as e:
            print(f"Error in ML utility evaluation: {str(e)}")
            print("Full error context:")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}