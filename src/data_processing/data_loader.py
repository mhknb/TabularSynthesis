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
        """Load data from uploaded file with optimized performance"""
        try:
            if filename.endswith(('.xls', '.xlsx')):
                # Use engine='openpyxl' which is more memory efficient for xlsx
                df = pd.read_excel(file, engine='openpyxl')
            elif filename.endswith('.csv'):
                # Use chunksize and low_memory for more efficient CSV processing
                # First check file size to determine if chunking is needed
                file_size = file.getbuffer().nbytes
                
                if file_size > 100 * 1024 * 1024:  # If file is larger than 100MB
                    chunks = pd.read_csv(file, chunksize=100000, low_memory=True)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file, low_memory=True)
            elif filename.endswith('.parquet'):
                df = pq.read_table(file).to_pandas()
            else:
                return None, "Unsupported file format. Please upload .xls, .xlsx, .csv, or .parquet files."
            
            # Optimize memory usage by downcasting numeric columns
            df = DataLoader.optimize_dataframe_memory(df)
            
            return df, None
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
            
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric columns"""
        # Convert integers to the smallest possible integer type
        for col in df.select_dtypes(include=['int64']).columns:
            col_min, col_max = df[col].min(), df[col].max()
            
            if col_min >= 0:  # Unsigned integers
                if col_max < 2**8:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 2**16:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 2**32:
                    df[col] = df[col].astype(np.uint32)
            else:  # Signed integers
                if col_min > -2**7 and col_max < 2**7:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    df[col] = df[col].astype(np.int32)
                    
        # Convert floats to float32 if possible
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
            
        return df
    
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