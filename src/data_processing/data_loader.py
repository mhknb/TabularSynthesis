import pandas as pd
import pyarrow.parquet as pq
from typing import Tuple, Optional
import io
import torch
import numpy as np

class DataLoader:
    """Handles loading data from various file formats and preparing for torch operations"""
    
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

    @staticmethod
    def prepare_torch_data(data: pd.DataFrame, batch_size: int) -> torch.utils.data.DataLoader:
        """Prepare data for torch operations with proper batch size handling"""
        try:
            # Convert to torch tensor
            tensor_data = torch.FloatTensor(data.values)

            # Adjust batch size if needed
            num_samples = len(data)
            if batch_size >= num_samples:
                batch_size = max(8, num_samples // 4)  # Ensure at least 4 batches

            # Create DataLoader with adjusted batch size
            return torch.utils.data.DataLoader(
                tensor_data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0  # Avoid multiprocessing issues with Streamlit
            )
        except Exception as e:
            raise RuntimeError(f"Error preparing torch data: {str(e)}")