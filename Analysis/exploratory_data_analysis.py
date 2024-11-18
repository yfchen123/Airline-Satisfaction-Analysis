import pandas as pd


# Perform basic EDA to understand the structure and distribution of the dataset.
# Plot distributions of key features using histograms, box plots, etc.
# Visualize relationships between features and identify correlations using heatmaps.
# Discuss key insights drawn from EDA and potential challenges with the dataset (e.g., class imbalance, highly correlated features).

def mergeData():
  
  train = pd.read_csv("../Airline Data/train.csv")
  test = pd.read_csv("../Airline Data/test.csv")

  combined_data = pd.concat([train, test], ignore_index=True)
  print(combined_data)



# Test Functions
# mergeData()
