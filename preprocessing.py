import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def visualize_missing_values(data):
    # Calculate missing values for every column
    missing_counts = data.isnull().sum()
    # Calculate percentages
    missing_percentages = (missing_counts / len(data) * 100).round(1)

    # Visualize the missing values
    plt.figure(figsize=(15, 8))
    msno.bar(data)
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
        ax.text(bar.get_x() + bar.get_width() / 2, height + (current_ymax * 0.05),
                f'Missing:\n {count}',
                ha='center', va='bottom', color='black', size=9)

    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.show()


def missing_values_handler(data):
    # Checking for the missing values
    """total_missing_values = data.isnull().sum().sum()
    print(f"Total number of missing values is: {total_missing_values}")"""

    # visualize_missing_values(data)

    # Amputate missing values
    arrival_delay = data['Arrival Delay in Minutes']

    # Note that this mean does not include the missing values, so we are good
    mean_arrival_delay = arrival_delay.mean()
    data['Arrival Delay in Minutes'] = arrival_delay.fillna(mean_arrival_delay)

    # Check if there are any missing values after filling
    '''total_missing_values_after = data['Arrival Delay in Minutes'].isnull().sum()
    print(f"Total number of missing values after filling: {total_missing_values_after}")'''


def normalize_std(data):
    # Leave the ordinal features alone and standardize the rest of the numeric features.
    ordinal_features = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
                        "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
                        "On-board service", "Leg room service", "Baggage handling", "Checkin service",
                        "Inflight service",
                        "Cleanliness"]
    standard_features = ["Age", "Flight Distance", "Arrival Delay in Minutes", "Departure Delay in Minutes"]

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_features),  # Standardize the numerical features
            ('passthrough', 'passthrough', ordinal_features)  # Pass through the ordinal features without changes
        ])

    # Apply the transformations to your dataset
    data[standard_features + ordinal_features] = preprocessor.fit_transform(data[standard_features + ordinal_features])


def handle_categorical(data):
    """We only have two values for Gender, Customer Type, and Type of travel so we do
    label encoding for binary."""
    data['Gender'] = data['Gender'].map({"Male": 1, "Female": 0})
    data['Customer Type'] = data['Customer Type'].map({"Loyal Customer": 1, "disloyal Customer": 0})
    data['Type of Travel'] = data['Type of Travel'].map({"Business travel": 1, "Personal Travel": 0})

    '''Class has 3 values instead of 2 and so using label encoding does work as well, hence we do one 
    hot encoding.
    '''
    data = pd.get_dummies(data, columns=['Class'], dtype=int)


def preprocess(dataset):
    # I have decided to drop the ID column because it is a unique identifier that does not help with training.
    dataset.drop(columns=["id", "Unnamed: 0"], inplace=True, errors="ignore")

    # These are print statements to check the data
    # print(f"The training data is: {dataset}")

    # Handle the missing values
    missing_values_handler(dataset)

    # Normalize and standardize numerical features
    normalize_std(dataset)

    # Use one-hot encoding or label encoding to handle categorical attributes
    handle_categorical(dataset)
