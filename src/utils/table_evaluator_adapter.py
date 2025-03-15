"""
Adapter module for integrating table-evaluator functionality with enhanced type checking
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TableEvaluatorAdapter:
    """Adapter class to integrate table-evaluator functionality"""

    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cat_cols: Optional[List[str]] = None):
        """Initialize with real and synthetic data"""
        print("\nDEBUG - TableEvaluatorAdapter initialization:")
        print(f"Real data shape: {real_data.shape}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print(f"Real data columns: {real_data.columns.tolist()}")
        print(f"Synthetic data columns: {synthetic_data.columns.tolist()}")
        print("\nInitial data types:")
        print("Real data types:\n", real_data.dtypes)
        print("\nSynthetic data types:\n", synthetic_data.dtypes)

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

        # Initialize table-evaluator with processed data
        self._initialize_evaluator(cat_cols)

    def _initialize_evaluator(self, cat_cols: Optional[List[str]] = None):
        """Initialize table-evaluator with properly processed data"""
        try:
            # Identify categorical columns
            self.cat_cols = cat_cols or self._infer_categorical_columns()
            print(f"\nCategorical columns: {self.cat_cols}")

            # Process data
            self.real_processed = self.real_data.copy()
            self.synthetic_processed = self.synthetic_data.copy()

            # Handle categorical columns first
            for col in self.cat_cols:
                print(f"\nProcessing categorical column: {col}")
                try:
                    # Combine values from both datasets
                    all_values = pd.concat([self.real_processed[col], 
                                         self.synthetic_processed[col]]).astype(str)

                    # Create and fit encoder
                    encoder = LabelEncoder()
                    encoder.fit(all_values)

                    # Transform to float64
                    self.real_processed[col] = encoder.transform(
                        self.real_processed[col].astype(str)).astype(np.float64)
                    self.synthetic_processed[col] = encoder.transform(
                        self.synthetic_processed[col].astype(str)).astype(np.float64)

                    print(f"Encoded {col} - dtype: {self.real_processed[col].dtype}")
                except Exception as e:
                    print(f"Error encoding {col}: {str(e)}")
                    raise

            # Handle numeric columns
            numeric_cols = [col for col in self.real_processed.columns 
                          if col not in self.cat_cols]

            for col in numeric_cols:
                print(f"\nProcessing numeric column: {col}")
                try:
                    # Convert to numeric and handle missing values
                    self.real_processed[col] = pd.to_numeric(
                        self.real_processed[col], errors='coerce').fillna(0)
                    self.synthetic_processed[col] = pd.to_numeric(
                        self.synthetic_processed[col], errors='coerce').fillna(0)

                    # Scale to [0,1] range and convert to float64
                    scaler = StandardScaler()
                    self.real_processed[col] = scaler.fit_transform(
                        self.real_processed[col].values.reshape(-1, 1)).astype(np.float64).ravel()
                    self.synthetic_processed[col] = scaler.transform(
                        self.synthetic_processed[col].values.reshape(-1, 1)).astype(np.float64).ravel()

                    print(f"Processed {col} - dtype: {self.real_processed[col].dtype}")
                except Exception as e:
                    print(f"Error processing {col}: {str(e)}")
                    raise

            # Final type verification
            print("\nFinal data types:")
            print("Real processed types:\n", self.real_processed.dtypes)
            print("Synthetic processed types:\n", self.synthetic_processed.dtypes)

            # Verify no object types remain
            for df_name, df in [("Real", self.real_processed), 
                              ("Synthetic", self.synthetic_processed)]:
                obj_cols = df.select_dtypes(include=['object']).columns
                if not obj_cols.empty:
                    raise ValueError(f"{df_name} data still contains object columns: {obj_cols.tolist()}")

            # Initialize table-evaluator
            self.evaluator = TableEvaluator(
                real=self.real_processed,
                fake=self.synthetic_processed,
                cat_cols=self.cat_cols
            )
            print("TableEvaluator initialized successfully")

        except Exception as e:
            print(f"\nError in table-evaluator initialization: {str(e)}")
            print("\nDetailed data inspection:")
            for col in self.real_processed.columns:
                print(f"\nColumn: {col}")
                print(f"Real type: {self.real_processed[col].dtype}")
                print(f"Synthetic type: {self.synthetic_processed[col].dtype}")
                print(f"Real values: {self.real_processed[col].head()}")
                print(f"Synthetic values: {self.synthetic_processed[col].head()}")
            raise

    def _infer_categorical_columns(self) -> List[str]:
        """Infer categorical columns based on data types and unique values"""
        categorical_columns = []
        for col in self.real_data.columns:
            if self.real_data[col].dtype in ['object', 'category']:
                categorical_columns.append(col)
            else:
                n_unique = self.real_data[col].nunique()
                if n_unique < 50:  # Consider columns with less than 50 unique values as categorical
                    categorical_columns.append(col)
        return categorical_columns

    def evaluate_all(self, target_col: str) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        try:
            print(f"\nRunning evaluation with target column: {target_col}")
            print(f"Target column type - Real: {self.real_processed[target_col].dtype}")
            print(f"Target column type - Synthetic: {self.synthetic_processed[target_col].dtype}")

            ml_scores = self.evaluator.evaluate(target_col=target_col)
            print("Evaluation completed successfully")

            return {
                'classifier_scores': ml_scores.get('Classifier F1-scores', None),
                'privacy': {
                    'Duplicate rows between sets (real/fake)': 
                        ml_scores.get('Duplicate rows between sets (real/fake)', (0, 0)),
                    'nearest neighbor mean': ml_scores.get('nearest neighbor mean', 0),
                    'nearest neighbor std': ml_scores.get('nearest neighbor std', 0)
                },
                'correlation': {
                    'Column Correlation Distance RMSE': 
                        ml_scores.get('Column Correlation Distance RMSE', 0),
                    'Column Correlation distance MAE': 
                        ml_scores.get('Column Correlation distance MAE', 0)
                },
                'similarity': {
                    'basic statistics': ml_scores.get('basic statistics', 0),
                    'Correlation column correlations': 
                        ml_scores.get('Correlation column correlations', 0),
                    'Mean Correlation between fake and real columns': 
                        ml_scores.get('Mean Correlation between fake and real columns', 0),
                    '1 - MAPE Estimator results': ml_scores.get('1 - MAPE Estimator results', 0),
                    '1 - MAPE 5 PCA components': ml_scores.get('1 - MAPE 5 PCA components', 0),
                    'Similarity Score': ml_scores.get('Similarity Score', 0)
                }
            }

        except Exception as e:
            print(f"\nError in evaluation: {str(e)}")
            print("\nData state at error:")
            print(f"Target column '{target_col}' details:")
            print(f"Real values: {self.real_processed[target_col].head()}")
            print(f"Synthetic values: {self.synthetic_processed[target_col].head()}")
            raise

    def get_visual_evaluation(self):
        """Generate visual evaluation plots"""
        try:
            print("\nGenerating visual evaluation plots")
            return self.evaluator.visual_evaluation()
        except Exception as e:
            print(f"Error in visual evaluation: {str(e)}")
            raise