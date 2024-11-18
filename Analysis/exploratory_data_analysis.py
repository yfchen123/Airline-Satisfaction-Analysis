import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Perform basic EDA to understand the structure and distribution of the dataset.
# Plot distributions of key features using histograms, box plots, etc.
# Visualize relationships between features and identify correlations using heatmaps.
# Discuss key insights drawn from EDA and potential challenges with the dataset (e.g., class imbalance, highly correlated features).

def merge_data():
  
  train = pd.read_csv("../Airline Data/train.csv")
  test = pd.read_csv("../Airline Data/test.csv")

  combined_data = pd.concat([train, test], ignore_index=True)
  
  return combined_data

def class_satisfaction_barplot(data):
  # This function returns a barplot of the three classes and how many satisfied customers per class.
  
  # Get class and satisfaction columns where satisfaction = 'satisfied'.
  class_data = data[['Class', 'satisfaction']]
  class_data = class_data[class_data['satisfaction'] == 'satisfied']
  
  # Get satisfaction counts of each category of class
  eco_count = len(class_data[class_data['Class'] == 'Eco'])
  eco_plus_count = len(class_data[class_data['Class'] == 'Eco Plus'])
  business_count = len(class_data[class_data['Class'] == 'Business'])

  # Get class columns where satisfaction = 'neutral or dissatisfied'
  class_data_negative = data[['Class', 'satisfaction']]
  class_data_negative = class_data_negative[class_data_negative['satisfaction'] == 'neutral or dissatisfied']
  
  # Get disatisfaction counts of each category of class
  eco_count_negative = len(class_data_negative[class_data_negative['Class'] == 'Eco'])
  eco_plus_count_negative = len(class_data_negative[class_data_negative['Class'] == 'Eco Plus'])
  business_count_negative = len(class_data_negative[class_data_negative['Class'] == 'Business'])

  # New df for counts:
  counts = {

        'Class': ['Eco', 'Eco Plus', 'Business','Eco', 'Eco Plus', 'Business'],
        'Count': [eco_count, eco_plus_count, business_count, eco_count_negative, eco_plus_count_negative, business_count_negative],
        'Satisfied': ['satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied','neutral or dissatisfied','neutral or dissatisfied']
        }
  
  counts = pd.DataFrame(counts)
  
  # Plot (using Seaborn)
  
  plot = sns.catplot(data=counts, kind='bar', x='Class', y='Count', hue='Satisfied', palette='viridis',
                      height=6, aspect=1.2, legend=False)
  plot.set_axis_labels("Class", "Count")
  plt.legend(loc='upper right')
  plt.savefig('./Figures/class_satisfaction_barplot.png')







data = merge_data()

class_satisfaction_barplot(data)
