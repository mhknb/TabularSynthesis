"""
Test script to verify the optimizations made to data processing and evaluation components
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from src.data_processing.data_loader import DataLoader
from src.data_processing.transformers import DataTransformer

def test_data_transformer_optimizations():
    """Test the optimized DataTransformer with caching"""
    print("Testing DataTransformer optimizations...")
    
    # Create a large sample dataframe
    n_rows = 100000
    print(f"Creating sample dataframe with {n_rows} rows")
    
    # Generate continuous data
    np.random.seed(42)
    continuous_data = pd.DataFrame({
        'col1': np.random.normal(0, 1, n_rows),
        'col2': np.random.exponential(2, n_rows),
        'col3': np.random.uniform(0, 100, n_rows)
    })
    
    # Generate categorical data
    categories = ['A', 'B', 'C', 'D', 'E']
    categorical_data = pd.DataFrame({
        'cat1': np.random.choice(categories, n_rows),
        'cat2': np.random.choice(['X', 'Y', 'Z'], n_rows),
    })
    
    # Combine data
    data = pd.concat([continuous_data, categorical_data], axis=1)
    
    # Create transformer
    transformer = DataTransformer()
    
    # Test continuous transformation with caching
    print("\nTesting continuous transformation with caching...")
    
    # First run - should be slower
    start_time = time.time()
    transformed_cont1 = transformer.transform_continuous(data['col1'])
    first_run_time = time.time() - start_time
    print(f"First run time: {first_run_time:.4f} seconds")
    
    # Second run - should use cache and be faster
    start_time = time.time()
    transformed_cont1_again = transformer.transform_continuous(data['col1'])
    second_run_time = time.time() - start_time
    print(f"Second run time: {second_run_time:.4f} seconds")
    print(f"Speedup: {first_run_time / second_run_time:.2f}x")
    
    # Test categorical transformation with caching
    print("\nTesting categorical transformation with caching...")
    
    # First run - should be slower
    start_time = time.time()
    transformed_cat1 = transformer.transform_categorical(data['cat1'])
    first_run_time = time.time() - start_time
    print(f"First run time: {first_run_time:.4f} seconds")
    
    # Second run - should use cache and be faster
    start_time = time.time()
    transformed_cat1_again = transformer.transform_categorical(data['cat1'])
    second_run_time = time.time() - start_time
    print(f"Second run time: {second_run_time:.4f} seconds")
    print(f"Speedup: {first_run_time / second_run_time:.2f}x")
    
    return True

def test_data_loader_optimizations():
    """Test the DataLoader memory optimization functionality"""
    print("\nTesting DataLoader memory optimizations...")
    
    # Create a sample dataset with different numeric types
    n_rows = 10000
    data = pd.DataFrame({
        'small_ints': np.random.randint(0, 100, n_rows),  # Should convert to uint8
        'med_ints': np.random.randint(-1000, 1000, n_rows),  # Should convert to int16
        'large_ints': np.random.randint(0, 100000, n_rows),  # Should convert to uint32
        'floats': np.random.random(n_rows) * 100  # Should convert to float32
    })
    
    # Get memory usage before optimization
    before_size = data.memory_usage(deep=True).sum()
    print(f"Memory usage before optimization: {before_size / 1024:.2f} KB")
    
    # Optimize memory
    optimized_data = DataLoader.optimize_dataframe_memory(data)
    
    # Get memory usage after optimization
    after_size = optimized_data.memory_usage(deep=True).sum()
    print(f"Memory usage after optimization: {after_size / 1024:.2f} KB")
    print(f"Memory reduction: {(1 - after_size / before_size) * 100:.2f}%")
    
    # Verify data types
    print("\nData types after optimization:")
    for col in optimized_data.columns:
        print(f"{col}: {optimized_data[col].dtype}")
    
    return True

def main():
    """Run the optimization tests"""
    print("=== Testing Performance Optimizations ===\n")
    
    # Test DataTransformer optimizations
    transformer_result = test_data_transformer_optimizations()
    
    # Test DataLoader optimizations
    loader_result = test_data_loader_optimizations()
    
    if transformer_result and loader_result:
        print("\n✅ All optimization tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the logs.")

if __name__ == "__main__":
    main()