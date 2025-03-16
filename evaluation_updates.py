"""
Temporary file to hold the updated functions for evaluation.py
"""

def generate_evaluation_plots(self):
    """Generate evaluation plots with enhanced error handling"""
    try:
        print("\nGenerating evaluation plots...")
        figures = []

        # Get numeric and categorical columns
        numeric_cols = self.real_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.real_data.select_dtypes(include=['object', 'category']).columns

        n_cols = len(categorical_cols)
        if n_cols == 0:
            print("No categorical columns found")
            return None

        print(f"Processing {n_cols} categorical columns")

        # Group columns by feature type based on common prefixes or patterns in column names
        column_groups = {}

        try:
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
            for group_name, cols in column_groups.items():
                n_cols_in_group = len(cols)
                if n_cols_in_group > 1:
                    # Determine grid layout based on number of columns in this group
                    if n_cols_in_group <=3:
                        fig, axes = plt.subplots(1, n_cols_in_group, figsize=(6*n_cols_in_group, 6))
                    else:
                        n_rows = (n_cols_in_group + 2) // 3  # Up to 3 columns per row
                        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
                        if n_rows == 1:
                            axes = axes.reshape(1, -1)
                    flat_axes = axes.flatten()
                    for i, col in enumerate(cols):
                        try:
                            ax = flat_axes[i]
                            real_counts = self.real_data[col].value_counts(normalize=True)
                            synth_counts = self.synthetic_data[col].value_counts(normalize=True)
                            if real_counts.empty or synth_counts.empty:
                                print(f"Warning: Empty counts for column {col}")
                                continue

                            all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
                            real_values = np.zeros(len(all_categories))
                            synth_values = np.zeros(len(all_categories))

                            for j, category in enumerate(all_categories):
                                real_values[j] = real_counts.get(category, 0)
                                synth_values[j] = synth_counts.get(category, 0)
                            real_cumsum = np.cumsum(real_values)
                            synth_cumsum = np.cumsum(synth_values)
                            
                            # Updated plotting style: scatter plots with no connecting lines
                            ax.plot(range(len(all_categories)), real_cumsum, linestyle='none', marker='o', 
                                 label='Real', color='darkblue', markersize=8)
                            ax.plot(range(len(all_categories)), synth_cumsum, linestyle='none', marker='o', 
                                 label='Fake', color='sandybrown', markersize=8, alpha=0.7)
                            
                            ax.set_ylim(0, 1.05)
                            ax.set_xlim(-0.5, len(all_categories) - 0.5)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.set_title(col)
                            ax.set_xlabel('')
                            ax.set_ylabel('Cumsum')
                            if len(all_categories) <= 10:
                                ax.set_xticks(range(len(all_categories)))
                                ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=8)
                            else:
                                step = max(1, len(all_categories) // 5)
                                indices = range(0, len(all_categories), step)
                                categories = [all_categories[i] for i in indices]
                                ax.set_xticks(indices)
                                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
                            if i == 0:
                                ax.legend(loc='upper left')
                        except Exception as e:
                            print(f"Error plotting column {col}: {str(e)}")
                            continue

                    for j in range(n_cols_in_group, len(flat_axes)):
                        flat_axes[j].set_visible(False)

                    fig.suptitle(f"{group_name}", fontsize=16)
                    plt.tight_layout()
                    fig.subplots_adjust(top=0.9)
                    fig_list.append(fig)
                    if save_path:
                        fig.savefig(f"{save_path}_{group_name}.png")
                        plt.close(fig)
                elif n_cols_in_group == 1:
                    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                    axes = np.array([axes])
                    for i, col in enumerate(cols):
                        try:
                            ax = axes[0]
                            real_counts = self.real_data[col].value_counts(normalize=True)
                            synth_counts = self.synthetic_data[col].value_counts(normalize=True)
                            if real_counts.empty or synth_counts.empty:
                                print(f"Warning: Empty counts for column {col}")
                                continue
                            all_categories = sorted(set(real_counts.index) | set(synth_counts.index))
                            real_values = np.zeros(len(all_categories))
                            synth_values = np.zeros(len(all_categories))
                            for j, category in enumerate(all_categories):
                                real_values[j] = real_counts.get(category, 0)
                                synth_values[j] = synth_counts.get(category, 0)
                            real_cumsum = np.cumsum(real_values)
                            synth_cumsum = np.cumsum(synth_values)
                            
                            # Updated plotting style: scatter plots with no connecting lines
                            ax.plot(range(len(all_categories)), real_cumsum, linestyle='none', marker='o', 
                                 label='Real', color='darkblue', markersize=8)
                            ax.plot(range(len(all_categories)), synth_cumsum, linestyle='none', marker='o', 
                                 label='Fake', color='sandybrown', markersize=8, alpha=0.7)
                            
                            ax.set_ylim(0, 1.05)
                            ax.set_xlim(-0.5, len(all_categories) - 0.5)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.set_title(col)
                            ax.set_xlabel('')
                            ax.set_ylabel('Cumsum')
                            if len(all_categories) <= 10:
                                ax.set_xticks(range(len(all_categories)))
                                ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=8)
                            else:
                                step = max(1, len(all_categories) // 5)
                                indices = range(0, len(all_categories), step)
                                categories = [all_categories[i] for i in indices]
                                ax.set_xticks(indices)
                                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
                            ax.legend(loc='upper left')
                        except Exception as e:
                            print(f"Error plotting column {col}: {str(e)}")
                            continue
                    fig_list.append(fig)
                    if save_path:
                        fig.savefig(f"{save_path}_{group_name}.png")
                        plt.close(fig)
            figures.extend(fig_list)
        except Exception as e:
            print(f"Error processing categorical columns: {str(e)}")

        # Get numeric columns
        numeric_cols = self.real_data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            # Create figure for mean and std plots
            fig_mean_std = plt.figure(figsize=(12, 6))
            ax = fig_mean_std.add_subplot(111)

            # Calculate statistics safely
            real_means = np.log(np.abs(self.real_data[numeric_cols].mean() + 1e-10))
            synth_means = np.log(np.abs(self.synthetic_data[numeric_cols].mean() + 1e-10))
            real_stds = np.log(self.real_data[numeric_cols].std() + 1e-10)
            synth_stds = np.log(self.synthetic_data[numeric_cols].std() + 1e-10)

            x = range(len(numeric_cols))
            width = 0.35

            # Plot bars
            ax.bar([i - width/2 for i in x], real_means, width, label='Real Mean', color='blue', alpha=0.5)
            ax.bar([i + width/2 for i in x], synth_means, width, label='Synthetic Mean', color='red', alpha=0.5)
            ax.bar([i - width/2 for i in x], real_stds, width, bottom=real_means, label='Real Std', color='blue', alpha=0.3)
            ax.bar([i + width/2 for i in x], synth_stds, width, bottom=synth_means, label='Synthetic Std', color='red', alpha=0.3)

            ax.set_xticks(x)
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax.set_title('Absolute Log Mean and STDs of numeric data')
            ax.legend()
            plt.tight_layout()
            figures.append(fig_mean_std)

            # Create figure for cumulative sums of numerical data
            fig_num_cumsums = plt.figure(figsize=(15, 10))
            cols_per_row = min(4, len(numeric_cols))
            rows = (len(numeric_cols) - 1) // cols_per_row + 1

            for i, col in enumerate(numeric_cols):
                ax = fig_num_cumsums.add_subplot(rows, cols_per_row, i + 1)

                # Calculate CDFs efficiently
                real_col = np.sort(self.real_data[col].values)
                synth_col = np.sort(self.synthetic_data[col].values)

                # Use fewer points for smoother plotting
                n_points = min(1000, len(real_col))
                indices = np.linspace(0, len(real_col)-1, n_points).astype(int)

                real_cdf = np.arange(1, len(real_col) + 1) / len(real_col)
                synth_cdf = np.arange(1, len(synth_col) + 1) / len(synth_col)

                # Plot CDFs using subset of points
                ax.plot(real_col[indices], real_cdf[indices], label='Real', color='blue')
                ax.plot(synth_col[indices], synth_cdf[indices], label='Synthetic', color='red')
                ax.set_title(col)
                ax.set_xlabel('Value')
                ax.set_ylabel('Cumulative Probability')
                ax.legend()

            plt.tight_layout()
            figures.append(fig_num_cumsums)

        # Generate categorical CDF plots
        categorical_cdfs = self.plot_categorical_cdf()
        if categorical_cdfs:
            figures.extend(categorical_cdfs)

        print("Successfully generated evaluation plots")
        return figures

    except Exception as e:
        print(f"Error generating evaluation plots: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return None