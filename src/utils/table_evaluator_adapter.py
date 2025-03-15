"""
Adapter module for integrating table-evaluator functionality
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator

class TableEvaluatorAdapter:
    """Adapter class to integrate table-evaluator functionality"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cat_cols: Optional[List[str]] = None):
        """
        Initialize the adapter with real and synthetic data
        """
        print("\nDEBUG - TableEvaluatorAdapter initialization:")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print(f"Real data columns: {real_data.columns.tolist()}")
        print(f"Synthetic data columns: {synthetic_data.columns.tolist()}")

        self.real_data = real_data.copy()
        self.synthetic_data = synthetic_data.copy()
        self.cat_cols = cat_cols or self._infer_categorical_columns()

        print(f"Categorical columns: {self.cat_cols}")

        # Remove duplicates from synthetic data
        orig_len = len(self.synthetic_data)
        self.synthetic_data = self.synthetic_data.drop_duplicates()
        dropped_dupes = orig_len - len(self.synthetic_data)
        print(f"Removed {dropped_dupes} duplicate rows from synthetic data")

        # Print data types for debugging
        print("\nDEBUG - Data types before preprocessing:")
        print("Real data types:")
        print(self.real_data.dtypes)
        print("\nSynthetic data types:")
        print(self.synthetic_data.dtypes)

        # Preprocess data
        self._preprocess_data()

        print("\nDEBUG - Data types after preprocessing:")
        print("Real data types:")
        print(self.real_data.dtypes)
        print("\nSynthetic data types:")
        print(self.synthetic_data.dtypes)

        try:
            self.evaluator = TableEvaluator(
                real=self.real_data,
                fake=self.synthetic_data,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")
        except Exception as e:
            print(f"Error initializing TableEvaluator: {str(e)}")
            print("Real data sample:")
            print(self.real_data.head())
            print("\nSynthetic data sample:")
            print(self.synthetic_data.head())
            raise

    def _preprocess_data(self):
        """Preprocess data to ensure compatible types"""
        # Handle numeric columns
        numeric_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if col not in self.cat_cols:
                # Convert to float64 for consistency
                self.real_data[col] = pd.to_numeric(self.real_data[col], errors='coerce').astype('float64')
                self.synthetic_data[col] = pd.to_numeric(self.synthetic_data[col], errors='coerce').astype('float64')

        # Handle categorical columns
        for col in self.cat_cols:
            # Convert to category type
            self.real_data[col] = self.real_data[col].astype('category')
            self.synthetic_data[col] = self.synthetic_data[col].astype('category')

        # Handle datetime columns
        datetime_cols = self.real_data.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            # Convert datetime to numeric timestamp
            self.real_data[col] = pd.to_datetime(self.real_data[col]).astype(np.int64) // 10**9
            self.synthetic_data[col] = pd.to_datetime(self.synthetic_data[col]).astype(np.int64) // 10**9

        # Handle object columns that aren't categorical
        object_cols = self.real_data.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col not in self.cat_cols:
                # Try to convert to numeric, if not possible make categorical
                try:
                    self.real_data[col] = pd.to_numeric(self.real_data[col], errors='coerce')
                    self.synthetic_data[col] = pd.to_numeric(self.synthetic_data[col], errors='coerce')
                except:
                    print(f"Converting column {col} to categorical")
                    self.cat_cols.append(col)
                    self.real_data[col] = self.real_data[col].astype('category')
                    self.synthetic_data[col] = self.synthetic_data[col].astype('category')

        # Fill any remaining NaN values
        self.real_data = self.real_data.fillna(0)
        self.synthetic_data = self.synthetic_data.fillna(0)

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on number of unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            # Check if column is already categorical or object type
            if col in self.real_data.select_dtypes(include=['object', 'category']).columns:
                categorical_columns.append(col)
            else:
                n_unique = self.real_data[col].nunique()
                if n_unique < 50:  # Consider columns with less than 50 unique values as categorical
                    categorical_columns.append(col)

        print(f"Inferred categorical columns: {categorical_columns}")
        return categorical_columns

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """
        Run comprehensive evaluation using table-evaluator
        """
        try:
            print("\nDEBUG - Running table-evaluator evaluation")
            print(f"Target column: {target_col}")

            # Verify target column exists and check its type
            print(f"Target column type in real data: {self.real_data[target_col].dtype}")
            print(f"Target column type in synthetic data: {self.synthetic_data[target_col].dtype}")

            # Convert target column to numeric if possible
            if target_col not in self.cat_cols:
                try:
                    self.real_data[target_col] = pd.to_numeric(self.real_data[target_col], errors='coerce')
                    self.synthetic_data[target_col] = pd.to_numeric(self.synthetic_data[target_col], errors='coerce')
                    print(f"Converted target column to numeric. New dtype: {self.real_data[target_col].dtype}")
                except Exception as e:
                    print(f"Could not convert target column to numeric: {str(e)}")
                    print("Adding to categorical columns")
                    if target_col not in self.cat_cols:
                        self.cat_cols.append(target_col)
                    self.real_data[target_col] = self.real_data[target_col].astype('category')
                    self.synthetic_data[target_col] = self.synthetic_data[target_col].astype('category')

            try:
                # Get evaluation results
                print("Starting evaluation...")
                ml_scores = self.evaluator.evaluate(target_col=target_col)
                print("Evaluation completed successfully")

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
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                print("Data sample causing the error:")
                print("Real data head:")
                print(self.real_data.head())
                print("\nSynthetic data head:")
                print(self.synthetic_data.head())
                raise

        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            print("Detailed error context:")
            import traceback
            traceback.print_exc()
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            print("\nDEBUG - Generating visual evaluation")
            plots = self.evaluator.visual_evaluation()
            print("Visual evaluation completed successfully")
            return plots
        except Exception as e:
            print(f"Error in visual evaluation: {str(e)}")
            raise