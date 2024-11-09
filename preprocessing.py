import missingno as msno
import matplotlib.pyplot as plt


def missing_values_handler(data):
    # Checking for the missing values
    total_missing_values = data.isnull().sum().sum()
    print(f"Total number of missing values is: {total_missing_values}")

    # Calculate missing values for every column
    missing_counts = data.isnull().sum()
    # Calculate percentages
    missing_percentages = (missing_counts / len(data) * 100).round(1)

    # Visualize the missing values
    plt.figure(figsize=(15, 8))
    msno_bar = msno.bar(data)
    plt.xticks(rotation=45, ha='right')
    plt.title("Missing Values by Column")

    # Get all axes in the figure
    fig = plt.gcf()
    axes = fig.get_axes()

    # Remove the second (top) axis completely
    if len(axes) > 1:
        fig.delaxes(axes[2])
        fig.delaxes(axes[1])

    # Keep main axis and its labels
    ax = axes[0]
    ax.yaxis.set_ticks([])

    # Extend y-axis limits to make room for labels
    current_ymax = ax.get_ylim()[1]
    ax.set_ylim(0, current_ymax * 1.2)  # Add 20% more space at top

    # Get the bars directly from the axis
    bars = [patch for patch in ax.patches if isinstance(patch, plt.Rectangle)]

    # Annotate each bar with both count and percentage
    for bar, count, pct in zip(bars, missing_counts, missing_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (current_ymax * 0.05),
                f'Missing:\n {count}',  # Just show count for now
                ha='center', va='bottom', color='black', size=9)

    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.show()


def preprocess(train_set):
    # These are print statements to check the data
    print(f"The training data is: {train_set}")

    # Handle the missing values
    missing_values_handler(train_set)
