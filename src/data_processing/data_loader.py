import pandas as pd
import pyarrow.parquet as pq
from typing import Tuple, Optional
import io

class DataLoader:
    """Handles loading data from various file formats"""
    
    @staticmethod
    def load_data(file: io.BytesIO, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load data from uploaded file"""
        try:
            if filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            elif filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith('.parquet'):
                df = pq.read_table(file).to_pandas()
            else:
                return None, "Unsupported file format. Please upload .xls, .xlsx, .csv, or .parquet files."
            
            return df, None
        except Exception as e:
            return None, f"Error loading file: {str(e)}"

    @staticmethod
    def infer_column_types(df: pd.DataFrame) -> dict:
        """Infer column types from DataFrame"""
        type_map = {}
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                type_map[column] = 'Continuous'
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                type_map[column] = 'Datetime'
            else:
                type_map[column] = 'Categorical'
        return type_map
