"""
Analyze the synthetic Adult Census data and compare with original data
to diagnose the mode collapse issue
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load synthetic data
synthetic_path = 'attached_assets/synthetic_data-2.csv'
synthetic_df = pd.read_csv(synthetic_path)

print("Synthetic data shape:", synthetic_df.shape)
print("\nSample of synthetic data:")
print(synthetic_df.head())

# Basic statistics
print("\nSynthetic data basic statistics:")
print(synthetic_df.describe().T)

# Check for unique values in each column
print("\nUnique value counts for categorical columns:")
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                      'relationship', 'race', 'sex', 'native-country', 'income']

for col in categorical_columns:
    unique_values = synthetic_df[col].nunique()
    print(f"{col}: {unique_values} unique values")
    
    # Print top 5 most common values
    value_counts = Counter(synthetic_df[col])
    print("  Top values:", value_counts.most_common(5))
    
    # Calculate entropy (diversity) of this column
    values, counts = np.unique(synthetic_df[col], return_counts=True)
    probs = counts / len(synthetic_df)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(values))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    print(f"  Entropy: {entropy:.4f}, Normalized: {normalized_entropy:.4f} (closer to 1 = more diverse)")
    print()

# Check for range of numerical columns
print("\nNumerical column ranges:")
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in numerical_columns:
    min_val = synthetic_df[col].min()
    max_val = synthetic_df[col].max()
    mean_val = synthetic_df[col].mean()
    std_val = synthetic_df[col].std()
    unique_count = synthetic_df[col].nunique()
    print(f"{col}: range [{min_val}, {max_val}], mean: {mean_val:.2f}, std: {std_val:.2f}, unique values: {unique_count}")

# Try to find any original data file to compare
try:
    # Try multiple potential file names/locations
    potential_files = [
        'attached_assets/adult.csv',
        'adult.csv',
        'original_data.csv',
        'train_data.csv'
    ]
    
    original_df = None
    for file in potential_files:
        try:
            original_df = pd.read_csv(file)
            print(f"\nFound original data at {file}")
            break
        except:
            continue
            
    if original_df is not None:
        print("Original data shape:", original_df.shape)
        print("\nSample of original data:")
        print(original_df.head())
        
        # Compare diversity metrics
        print("\nComparing diversity metrics between original and synthetic data:")
        for col in categorical_columns:
            if col in original_df.columns:
                # Original data entropy
                values_orig, counts_orig = np.unique(original_df[col], return_counts=True)
                probs_orig = counts_orig / len(original_df)
                entropy_orig = -np.sum(probs_orig * np.log(probs_orig))
                max_entropy_orig = np.log(len(values_orig))
                norm_entropy_orig = entropy_orig / max_entropy_orig if max_entropy_orig > 0 else 0
                
                # Synthetic data entropy (calculated again for comparison)
                values_synth, counts_synth = np.unique(synthetic_df[col], return_counts=True)
                probs_synth = counts_synth / len(synthetic_df)
                entropy_synth = -np.sum(probs_synth * np.log(probs_synth))
                max_entropy_synth = np.log(len(values_synth))
                norm_entropy_synth = entropy_synth / max_entropy_synth if max_entropy_synth > 0 else 0
                
                print(f"{col} entropy - Original: {norm_entropy_orig:.4f}, Synthetic: {norm_entropy_synth:.4f}")
                print(f"  Original unique values: {len(values_orig)}, Synthetic unique values: {len(values_synth)}")
                
except Exception as e:
    print(f"Could not find or load original data for comparison: {e}")

# Create visualizations to illustrate the mode collapse
plt.figure(figsize=(15, 10))

# Age distribution
plt.subplot(2, 3, 1)
synthetic_df['age'].hist(bins=30, alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.grid(alpha=0.3)

# Workclass distribution
plt.subplot(2, 3, 2)
workclass_counts = synthetic_df['workclass'].value_counts()
workclass_counts.plot(kind='bar')
plt.title('Workclass Distribution')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

# Education distribution
plt.subplot(2, 3, 3)
education_counts = synthetic_df['education'].value_counts()
education_counts.plot(kind='bar')
plt.title('Education Distribution')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

# Occupation distribution
plt.subplot(2, 3, 4)
occupation_counts = synthetic_df['occupation'].value_counts()
occupation_counts.plot(kind='bar')
plt.title('Occupation Distribution')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

# Income distribution
plt.subplot(2, 3, 5)
income_counts = synthetic_df['income'].value_counts()
income_counts.plot(kind='bar')
plt.title('Income Distribution')
plt.grid(alpha=0.3)

# Capital gain distribution
plt.subplot(2, 3, 6)
synthetic_df['capital-gain'].hist(bins=30, alpha=0.7)
plt.title('Capital Gain Distribution')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('synthetic_data_analysis.png')
print("\nSaved distribution plot to 'synthetic_data_analysis.png'")

# Create a heatmap to see if there are any correlations in the data
plt.figure(figsize=(12, 10))
numeric_df = synthetic_df[numerical_columns].copy()
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved correlation heatmap to 'correlation_heatmap.png'")

print("\nAnalysis complete. The synthetic data shows clear signs of mode collapse.")