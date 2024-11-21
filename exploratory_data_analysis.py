import matplotlib.pyplot as plt
import seaborn as sns
import math


def histogram_distribution(data):
    bins = 10
    columns = data.columns
    num_columns = len(columns)
    histograms_per_grid = 5

    # Split into groups of 5 columns each
    for i in range(0, num_columns, histograms_per_grid):
        grid_columns = columns[i:i + histograms_per_grid]
        num_plots = len(grid_columns)

        # Determine grid size for current group
        grid_cols = 3
        grid_rows = math.ceil(num_plots / grid_cols)

        # Create a grid of subplots
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 5 * grid_rows))
        axes = axes.flatten()

        for idx, column in enumerate(grid_columns):
            unique_vals = data[column].nunique()

            # Special handling for binary columns
            if unique_vals == 2:
                # Explicitly count occurrences of 0 and 1
                counts = data[column].value_counts().sort_index()
                axes[idx].bar(counts.index, counts.values, color='blue', alpha=0.7, edgecolor='black')
                axes[idx].set_xticks([0, 1])  # Set x-ticks to 0 and 1
            else:
                actual_bins = bins
                axes[idx].hist(data[column], bins=actual_bins, edgecolor='black', alpha=0.7)

            axes[idx].set_title(f'Histogram of {column}')
            axes[idx].set_xlabel(column)
            axes[idx].set_ylabel('Frequency')

        # Remove extra axes in the grid
        for idx in range(num_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(data, figsize=(15, 12)):
    # Compute the correlation matrix
    correlation_matrix = data.corr()

    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, cbar_kws={"shrink": 0.8}, square=True)

    # Add title and format ticks
    plt.title('Feature Correlation Heatmap', fontsize=18)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def visualize_data(data):
    # histogram_distribution(data)
    plot_correlation_heatmap(data)
