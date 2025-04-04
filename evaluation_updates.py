"""
Temporary file to hold the updated functions for evaluation.py
"""

def generate_evaluation_plots(self, save_path=None):
    """Generate evaluation plots with enhanced performance and error handling"""
    try:
        print("\nGenerating evaluation plots with optimized performance...")
        figures = []

        # Use parallel processing for large datasets
        use_parallel = max(len(self.real_data), len(self.synthetic_data)) > 100000
        
        # Get numeric and categorical columns
        numeric_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.real_data.select_dtypes(include=['object', 'category']).columns

        # Process categorical columns
        n_cols = len(categorical_cols)
        if n_cols > 0:
            print(f"Processing {n_cols} categorical columns")
            
            # Memory optimization: pre-calculate value counts for all categorical columns at once
            # This avoids redundant computations for each column
            print("Pre-computing categorical statistics...")
            real_counts_dict = {}
            synth_counts_dict = {}
            
            # Process columns in smaller batches to reduce memory usage
            batch_size = min(20, n_cols)
            for i in range(0, n_cols, batch_size):
                batch_cols = categorical_cols[i:i+batch_size]
                
                # Compute value counts for this batch
                for col in batch_cols:
                    try:
                        # For large datasets, sample for faster processing
                        if use_parallel and len(self.real_data) > 500000:
                            real_sample = self.real_data[col].sample(n=min(100000, len(self.real_data)))
                            synth_sample = self.synthetic_data[col].sample(n=min(100000, len(self.synthetic_data)))
                            real_counts_dict[col] = real_sample.value_counts(normalize=True)
                            synth_counts_dict[col] = synth_sample.value_counts(normalize=True)
                        else:
                            real_counts_dict[col] = self.real_data[col].value_counts(normalize=True)
                            synth_counts_dict[col] = self.synthetic_data[col].value_counts(normalize=True)
                    except Exception as e:
                        print(f"Error computing value counts for column {col}: {str(e)}")
                        continue
            
            # Group columns by feature type based on common prefixes or patterns in column names
            column_groups = {}
            
            # Check for different prefix patterns in column names
            for col in categorical_cols:
                # Look for common naming patterns like prefixes with underscores or other separators
                parts = col.split('_')
                if len(parts) > 1:
                    prefix = parts[0]
                    column_groups.setdefault(prefix, []).append(col)
                else:
                    # If no clear pattern, add to "other" group
                    column_groups.setdefault('other', []).append(col)

            fig_list = []
            
            # Process each group of related columns together
            print(f"Generating plots for {len(column_groups)} column groups...")
            for group_name, cols in column_groups.items():
                n_cols_in_group = len(cols)
                
                # Skip empty groups
                if n_cols_in_group == 0:
                    continue
                    
                # Process columns in multi-column layout
                if n_cols_in_group > 1:
                    # Determine optimal grid layout based on number of columns
                    if n_cols_in_group <= 3:
                        fig, axes = plt.subplots(1, n_cols_in_group, figsize=(6*n_cols_in_group, 6))
                    else:
                        # Use at most 4 columns per row for better readability
                        n_cols_per_row = min(4, n_cols_in_group)
                        n_rows = (n_cols_in_group + n_cols_per_row - 1) // n_cols_per_row
                        fig, axes = plt.subplots(n_rows, n_cols_per_row, figsize=(5*n_cols_per_row, 5*n_rows))
                        if n_rows == 1:
                            axes = axes.reshape(1, -1)
                    
                    # Flatten axes for easier indexing
                    flat_axes = axes.flatten()
                    
                    # Process each column
                    valid_cols_plotted = 0
                    for i, col in enumerate(cols):
                        try:
                            # Skip columns that weren't successfully counted
                            if col not in real_counts_dict or col not in synth_counts_dict:
                                continue
                                
                            real_counts = real_counts_dict[col]
                            synth_counts = synth_counts_dict[col]
                            
                            # Skip empty columns
                            if real_counts.empty or synth_counts.empty:
                                print(f"Warning: Empty counts for column {col}")
                                continue

                            # Get all unique categories across both datasets
                            all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
                            
                            # Limit to top 50 categories for very large category sets
                            if len(all_categories) > 50:
                                # Get top categories by frequency
                                top_real = set(real_counts.nlargest(25).index)
                                top_synth = set(synth_counts.nlargest(25).index)
                                all_categories = sorted(top_real | top_synth)
                                print(f"Column {col} has {len(real_counts.index)} categories, showing top {len(all_categories)}")
                                
                            # Pre-allocate arrays for better performance
                            real_values = np.zeros(len(all_categories))
                            synth_values = np.zeros(len(all_categories))

                            # Fill arrays with normalized counts
                            for j, category in enumerate(all_categories):
                                real_values[j] = real_counts.get(category, 0)
                                synth_values[j] = synth_counts.get(category, 0)
                                
                            # Compute cumulative sums
                            real_cumsum = np.cumsum(real_values)
                            synth_cumsum = np.cumsum(synth_values)
                            
                            # Get current axis
                            ax = flat_axes[valid_cols_plotted]
                            valid_cols_plotted += 1
                            
                            # Plot with optimized style
                            # Use scatter for fewer points, line for many points
                            if len(all_categories) <= 20:
                                ax.scatter(range(len(all_categories)), real_cumsum, 
                                          label='Real', color='darkblue', s=50, zorder=10)
                                ax.scatter(range(len(all_categories)), synth_cumsum, 
                                          label='Synthetic', color='sandybrown', s=50, alpha=0.7, zorder=5)
                            else:
                                # For many points, use lines instead of scatter for better performance
                                ax.plot(range(len(all_categories)), real_cumsum, 
                                       label='Real', color='darkblue', linewidth=2)
                                ax.plot(range(len(all_categories)), synth_cumsum, 
                                       label='Synthetic', color='sandybrown', linewidth=2, alpha=0.7)
                            
                            # Configure axis
                            ax.set_ylim(0, 1.05)
                            ax.set_xlim(-0.5, len(all_categories) - 0.5)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.set_title(col, fontsize=10)
                            ax.set_xlabel('')
                            ax.set_ylabel('Cumulative Probability')
                            
                            # Optimize tick labels based on number of categories
                            if len(all_categories) <= 10:
                                ax.set_xticks(range(len(all_categories)))
                                ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=8)
                            else:
                                # For many categories, show only a subset of ticks
                                max_ticks = 10
                                step = max(1, len(all_categories) // max_ticks)
                                indices = range(0, len(all_categories), step)
                                categories = [all_categories[i] for i in indices]
                                ax.set_xticks(indices)
                                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
                            
                            # Add legend only to first plot to avoid clutter
                            if valid_cols_plotted == 1:
                                ax.legend(loc='upper left', fontsize=8)
                        except Exception as e:
                            print(f"Error plotting column {col}: {str(e)}")
                            continue

                    # Hide unused axes
                    for j in range(valid_cols_plotted, len(flat_axes)):
                        flat_axes[j].set_visible(False)

                    # Set figure title
                    fig.suptitle(f"{group_name}", fontsize=16)
                    plt.tight_layout()
                    fig.subplots_adjust(top=0.92)
                    
                    # Add to figures list
                    fig_list.append(fig)
                    
                    # Save if path provided
                    if save_path:
                        filename = f"{save_path}_{group_name}.png"
                        try:
                            fig.savefig(filename)
                            plt.close(fig)
                            print(f"Saved plot to {filename}")
                        except Exception as e:
                            print(f"Error saving plot to {filename}: {str(e)}")
                
                # Process single-column groups
                elif n_cols_in_group == 1:
                    col = cols[0]
                    try:
                        # Skip columns that weren't successfully counted
                        if col not in real_counts_dict or col not in synth_counts_dict:
                            continue
                            
                        real_counts = real_counts_dict[col]
                        synth_counts = synth_counts_dict[col]
                        
                        # Skip empty columns
                        if real_counts.empty or synth_counts.empty:
                            print(f"Warning: Empty counts for column {col}")
                            continue
                            
                        # Create figure and axis
                        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                        
                        # Get all unique categories across both datasets
                        all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
                        
                        # Limit categories for large sets
                        if len(all_categories) > 50:
                            # Get top categories by frequency
                            top_real = set(real_counts.nlargest(25).index)
                            top_synth = set(synth_counts.nlargest(25).index)
                            all_categories = sorted(top_real | top_synth)
                            print(f"Column {col} has {len(real_counts.index)} categories, showing top {len(all_categories)}")
                            
                        # Pre-allocate arrays
                        real_values = np.zeros(len(all_categories))
                        synth_values = np.zeros(len(all_categories))
                        
                        # Fill arrays
                        for j, category in enumerate(all_categories):
                            real_values[j] = real_counts.get(category, 0)
                            synth_values[j] = synth_counts.get(category, 0)
                            
                        # Compute cumulative sums
                        real_cumsum = np.cumsum(real_values)
                        synth_cumsum = np.cumsum(synth_values)
                        
                        # Plot with optimized style
                        if len(all_categories) <= 20:
                            ax.scatter(range(len(all_categories)), real_cumsum, 
                                      label='Real', color='darkblue', s=80, zorder=10)
                            ax.scatter(range(len(all_categories)), synth_cumsum, 
                                      label='Synthetic', color='sandybrown', s=80, alpha=0.7, zorder=5)
                        else:
                            ax.plot(range(len(all_categories)), real_cumsum, 
                                   label='Real', color='darkblue', linewidth=3)
                            ax.plot(range(len(all_categories)), synth_cumsum, 
                                   label='Synthetic', color='sandybrown', linewidth=3, alpha=0.7)
                        
                        # Configure axis
                        ax.set_ylim(0, 1.05)
                        ax.set_xlim(-0.5, len(all_categories) - 0.5)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.set_title(col, fontsize=14)
                        ax.set_xlabel('')
                        ax.set_ylabel('Cumulative Probability')
                        
                        # Optimize tick labels
                        if len(all_categories) <= 20:
                            ax.set_xticks(range(len(all_categories)))
                            ax.set_xticklabels(all_categories, rotation=45, ha='right')
                        else:
                            max_ticks = 15
                            step = max(1, len(all_categories) // max_ticks)
                            indices = range(0, len(all_categories), step)
                            categories = [all_categories[i] for i in indices]
                            ax.set_xticks(indices)
                            ax.set_xticklabels(categories, rotation=45, ha='right')
                        
                        # Add legend
                        ax.legend(loc='upper left')
                        
                        # Adjust layout
                        plt.tight_layout()
                        
                        # Add to figures list
                        fig_list.append(fig)
                        
                        # Save if path provided
                        if save_path:
                            filename = f"{save_path}_{col}.png"
                            try:
                                fig.savefig(filename)
                                plt.close(fig)
                                print(f"Saved plot to {filename}")
                            except Exception as e:
                                print(f"Error saving plot to {filename}: {str(e)}")
                    except Exception as e:
                        print(f"Error creating single plot for column {col}: {str(e)}")
            
            # Add figures to main list
            figures.extend(fig_list)
        else:
            print("No categorical columns found")

        # Process numeric columns
        numeric_cols = self.real_data.select_dtypes(include=['float64', 'int64']).columns
        n_numeric = len(numeric_cols)
        
        if n_numeric > 0:
            print(f"Processing {n_numeric} numeric columns")
            
            # For datasets with many numeric columns, limit to most informative ones
            if n_numeric > 20:
                print(f"Too many numeric columns ({n_numeric}), selecting most informative ones")
                # Calculate variance ratios between real and synthetic data
                real_vars = self.real_data[numeric_cols].var()
                synth_vars = self.synthetic_data[numeric_cols].var()
                
                # Replace zeros with small values to avoid division by zero
                real_vars = real_vars.replace(0, 1e-10)
                synth_vars = synth_vars.replace(0, 1e-10)
                
                # Calculate ratios in both directions and take minimum to get columns with biggest differences
                ratios1 = real_vars / synth_vars
                ratios2 = synth_vars / real_vars
                diff_scores = pd.DataFrame({
                    'col': numeric_cols,
                    'score': np.maximum(ratios1, ratios2)
                })
                
                # Select top columns by difference score, at least 10 columns
                top_cols = diff_scores.nlargest(min(20, n_numeric), 'score')['col'].tolist()
                numeric_cols = top_cols
                print(f"Selected {len(numeric_cols)} most informative numeric columns")
            
            try:
                # Calculate statistics with better error handling and NA handling
                print("Computing numeric statistics...")
                
                # Function to safely calculate log values
                def safe_log(x):
                    return np.log(np.abs(x) + 1e-10)
                
                # Calculate statistics with NA handling
                real_means = safe_log(self.real_data[numeric_cols].mean())
                synth_means = safe_log(self.synthetic_data[numeric_cols].mean())
                real_stds = safe_log(self.real_data[numeric_cols].std() + 1e-10)
                synth_stds = safe_log(self.synthetic_data[numeric_cols].std() + 1e-10)
                
                # Create figure for mean and std plots
                print("Creating mean/std comparison plot...")
                fig_mean_std = plt.figure(figsize=(max(12, n_numeric//2), 8))
                ax = fig_mean_std.add_subplot(111)
                
                # Set up bar positions
                x = range(len(numeric_cols))
                width = 0.35
                
                # Plot bars with more distinct colors
                ax.bar([i - width/2 for i in x], real_means, width, label='Real Mean', color='#3498DB', alpha=0.7)
                ax.bar([i + width/2 for i in x], synth_means, width, label='Synthetic Mean', color='#E74C3C', alpha=0.7)
                ax.bar([i - width/2 for i in x], real_stds, width, bottom=real_means, label='Real Std', color='#2980B9', alpha=0.5)
                ax.bar([i + width/2 for i in x], synth_stds, width, bottom=synth_means, label='Synthetic Std', color='#C0392B', alpha=0.5)
                
                # Improve x-axis labels
                if len(numeric_cols) <= 10:
                    ax.set_xticks(x)
                    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                else:
                    # For many columns, show only a subset of ticks
                    step = max(1, len(numeric_cols) // 10)
                    indices = range(0, len(numeric_cols), step)
                    labels = [numeric_cols[i] for i in indices]
                    ax.set_xticks(indices)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # Add grid for better readability
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add title and legend
                ax.set_title('Comparison of Log Mean and Standard Deviation', fontsize=14)
                ax.legend(loc='upper right')
                
                # Adjust layout
                plt.tight_layout()
                
                # Add to figures list
                figures.append(fig_mean_std)
                
                # Save if path provided
                if save_path:
                    filename = f"{save_path}_numeric_stats.png"
                    try:
                        fig_mean_std.savefig(filename)
                        plt.close(fig_mean_std)
                        print(f"Saved numeric stats plot to {filename}")
                    except Exception as e:
                        print(f"Error saving numeric stats plot: {str(e)}")
                        
                # Create CDF plots for numeric columns
                print("Creating CDF plots for numeric columns...")
                cols_per_row = min(3, len(numeric_cols))
                rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
                
                fig_num_cumsums = plt.figure(figsize=(6*cols_per_row, 5*rows))
                
                # Process each column
                for i, col in enumerate(numeric_cols):
                    ax = fig_num_cumsums.add_subplot(rows, cols_per_row, i + 1)
                    
                    try:
                        # Get column values, handling NAs
                        real_col = self.real_data[col].dropna().values
                        synth_col = self.synthetic_data[col].dropna().values
                        
                        # Sort values for CDF calculation
                        real_col = np.sort(real_col)
                        synth_col = np.sort(synth_col)
                        
                        # Skip if either array is empty
                        if len(real_col) == 0 or len(synth_col) == 0:
                            print(f"Warning: Empty data for column {col}")
                            continue
                        
                        # Use fewer points for large datasets to improve performance
                        max_points = 500
                        if len(real_col) > max_points:
                            indices_real = np.linspace(0, len(real_col)-1, max_points).astype(int)
                            real_col_subset = real_col[indices_real]
                            real_cdf = np.linspace(0, 1, max_points)
                        else:
                            real_col_subset = real_col
                            real_cdf = np.arange(1, len(real_col) + 1) / len(real_col)
                            
                        if len(synth_col) > max_points:
                            indices_synth = np.linspace(0, len(synth_col)-1, max_points).astype(int)
                            synth_col_subset = synth_col[indices_synth]
                            synth_cdf = np.linspace(0, 1, max_points)
                        else:
                            synth_col_subset = synth_col
                            synth_cdf = np.arange(1, len(synth_col) + 1) / len(synth_col)
                        
                        # Plot CDFs
                        ax.plot(real_col_subset, real_cdf, label='Real', color='#3498DB', linewidth=2)
                        ax.plot(synth_col_subset, synth_cdf, label='Synthetic', color='#E74C3C', linewidth=2)
                        
                        # Configure axis
                        ax.set_title(col, fontsize=12)
                        ax.set_xlabel('Value', fontsize=10)
                        ax.set_ylabel('Cumulative Probability', fontsize=10)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Add legend
                        ax.legend(loc='best', fontsize=9)
                    except Exception as e:
                        print(f"Error plotting CDF for column {col}: {str(e)}")
                        continue
                
                # Adjust layout
                plt.tight_layout(pad=3.0)
                
                # Add to figures list
                figures.append(fig_num_cumsums)
                
                # Save if path provided
                if save_path:
                    filename = f"{save_path}_numeric_cdfs.png"
                    try:
                        fig_num_cumsums.savefig(filename)
                        plt.close(fig_num_cumsums)
                        print(f"Saved numeric CDFs plot to {filename}")
                    except Exception as e:
                        print(f"Error saving numeric CDFs plot: {str(e)}")
            except Exception as e:
                print(f"Error processing numeric columns: {str(e)}")
        else:
            print("No numeric columns found")

        print(f"Successfully generated {len(figures)} evaluation plots")
        return figures

    except Exception as e:
        print(f"Error generating evaluation plots: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return None