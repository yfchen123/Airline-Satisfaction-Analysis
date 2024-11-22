import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Perform basic EDA to understand the structure and distribution of the dataset.
# Plot distributions of key features using histograms, box plots, etc.
# Visualize relationships between features and identify correlations using heatmaps.
# Discuss key insights drawn from EDA and potential challenges with the dataset (e.g., class imbalance, highly correlated features).

def merge_data(fileA, fileB):
    print(f"merging {fileA} and {fileB}...")

    train = pd.read_csv(fileA)
    test = pd.read_csv(fileB)

    combined_data = pd.concat([train, test], ignore_index=True)

    print("data merged!")

    return combined_data


def class_analysis(data):
    # This function returns a barplot of the three classes and how many satisfied customers per class and
    # returns insights on age distribution among airlines classes.

    # Get Age, class and satisfaction columns where satisfaction = 'satisfied'.
    class_data = data.copy()
    class_data = class_data[['Age', 'Class', 'satisfaction']]
    class_data = class_data[class_data['satisfaction'] == 'satisfied']
    class_data.dropna()

    # Get satisfaction counts of each category of class
    eco_count = len(class_data[class_data['Class'] == 'Eco'])
    eco_plus_count = len(class_data[class_data['Class'] == 'Eco Plus'])
    business_count = len(class_data[class_data['Class'] == 'Business'])

    # Get class columns where satisfaction = 'neutral or dissatisfied'
    class_data_negative = data.copy()
    class_data_negative = class_data_negative[['Class', 'satisfaction']]
    class_data_negative = class_data_negative[class_data_negative['satisfaction'] == 'neutral or dissatisfied']

    # Get disatisfaction counts of each category of class
    eco_count_negative = len(class_data_negative[class_data_negative['Class'] == 'Eco'])
    eco_plus_count_negative = len(class_data_negative[class_data_negative['Class'] == 'Eco Plus'])
    business_count_negative = len(class_data_negative[class_data_negative['Class'] == 'Business'])

    # Construct Datastructure for barplot
    counts = {

        'Class': ['Eco', 'Eco Plus', 'Business', 'Eco', 'Eco Plus', 'Business'],
        'Count': [eco_count, eco_plus_count, business_count, eco_count_negative, eco_plus_count_negative,
                  business_count_negative],
        'Satisfied': ['satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied',
                      'neutral or dissatisfied']
    }

    # construct dataframe for counts
    counts = pd.DataFrame(counts)

    # Plot barplot of class and satisfaction counts (using Seaborn)
    plot = sns.catplot(data=counts, kind='bar', x='Class', y='Count', hue='Satisfied', palette='viridis',
                       height=6, aspect=1.2, legend=False)
    plot.set_axis_labels("Class", "Count")
    plot.fig.suptitle('class satisfaction barplot')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    plt.savefig('./Figures/class_satisfaction_barplot.png')

    print("Saved class_satisfaction_barplot.png in Analysis/Figures/")

    # making histograms of different airline class's age distributions (and also collecting data for ANOVA)
    x1 = class_histogram(class_data[class_data['Class'] == 'Eco'], field="Age", colour='purple',
                         classname='Economy', filename='economy_age_dist.png')

    x2 = class_histogram(class_data[class_data['Class'] == 'Eco Plus'], field="Age", colour='coral',
                         classname='Economy Plus', filename='economy_plus_age_dist.png')

    x3 = class_histogram(class_data[class_data['Class'] == 'Business'], field="Age", colour='pink',
                         classname='Business', filename='business_age_dist.png')

    # we can run an anova test to check whether there is a significant difference of means between
    # any of the age distributions among the classes.
    anova = stats.f_oneway(x1, x2, x3)
    print(f'anova p-value = {anova.pvalue}')

    # the results were significant, so we run post hoc analysis using tukey test.
    # first we have to "melt" the results.
    x = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
    x = x.fillna(value=-1)

    tukey_melt = pd.melt(x)
    tukey_melt = tukey_melt[tukey_melt['value'] != -1]

    tukey_test = pairwise_tukeyhsd(tukey_melt['value'], tukey_melt['variable'], alpha=0.05)

    print(tukey_test)


def distance_and_delay(data):
    # This function determines whether there is some sort of correlation between flight distance and delays and
    # infers on their standalone distributions.

    # Get the data for Flight Distancce, Arrival Delay in Minutes, satisfaction columns
    distance_delay = data[['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'satisfaction']]

    # Drop the results with delay = 0
    distance_delay = distance_delay[distance_delay['Arrival Delay in Minutes'] > 0]
    distance_delay = distance_delay[distance_delay['Departure Delay in Minutes'] == 0]
    distance_delay = distance_delay.dropna()

    # plot and find correlation distance vs time
    scatter_and_correlate(distance_delay, 'blue', 'Flight distance vs arrival delay', 'distance_delay_scatterplot.png')

    # Split data into satisfied and neutral/disatisfied groups
    positive = distance_delay[distance_delay['satisfaction'] == 'satisfied']
    negative = distance_delay[distance_delay['satisfaction'] != 'satisfied']

    scatter_and_correlate(positive, 'green', "Flight distance vs arrival delay of Satisfied Customers",
                          'distance_delay_satifised_scatterplot.png')
    scatter_and_correlate(negative, 'red', "flight distance vs arrival delay of Non-Satisfied Customers",
                          'distance_delay_dissatifised_scatterplot.png')

    # running histograms to check the flight distance distributions of satisfied and non satisfied customers are different
    class_histogram(d=positive, field='Flight Distance', colour='orange', classname='Satisfied',
                    filename='distance_distribution_satisfied.png', num_bins=100)
    class_histogram(d=negative, field='Flight Distance', colour='yellow', classname='Non-Satisfied',
                    filename='distance_distribution_nonsatisfied.png', num_bins=100)

    feature = 'Flight Distance'
    print(f'\nSatisfied customer flight distance mean = {positive[feature].mean()}')
    print(f'Non-satisfied customer flight distance mean = {negative[feature].mean()}')


def heatmapping(data, title, colour, filename):
    # This function returns a heatmap of specified data
    # get the correlation matrix
    corr_data = data.copy()
    corr_data = corr_data.corr()

    # plot the thingy
    plt.figure(figsize=(15, 13))
    sns.heatmap(corr_data, annot=True, cmap=colour, fmt=".2f", linewidths=0.4)
    plt.title(title)
    plt.savefig(f'./Figures/{filename}')


def type_of_travel(data):
    # this function does analysis on the type of travel again satisfaction.
    travel_data = data.copy()
    travel_data = travel_data[['satisfaction', 'Type of Travel']].dropna()

    # split into satisfied and nonsatisfied dataframes
    travel_satisfied = travel_data[travel_data['satisfaction'] == 'satisfied']
    travel_nonsatisfied = travel_data[travel_data['satisfaction'] != 'satisfied']

    # Get personal and business satisfied counts
    business_count_satisfied = len(travel_satisfied[travel_satisfied['Type of Travel'] == 'Business travel'])
    personal_count_satisfied = len(travel_satisfied[travel_satisfied['Type of Travel'] != 'Business travel'])

    # Get person and business non satisfied counts
    business_count_nonsatisfied = len(travel_nonsatisfied[travel_nonsatisfied['Type of Travel'] == 'Business travel'])
    personal_count_nonsatisfied = len(travel_nonsatisfied[travel_nonsatisfied['Type of Travel'] != 'Business travel'])

    # Construct Datastructure for barplot
    counts = {

        'Travel Type': ['Business travel', 'Personal Travel', 'Business travel', 'Personal Travel'],
        'Count': [business_count_satisfied, personal_count_satisfied, business_count_nonsatisfied,
                  personal_count_nonsatisfied],
        'Satisfied': ['satisfied', 'satisfied', 'neutral or dissatisfied', 'neutral or dissatisfied']
    }

    # Turn it into a DataFrame
    counts = pd.DataFrame(counts)

    plot = sns.catplot(data=counts, kind='bar', x='Travel Type', y='Count', hue='Satisfied', palette='mako',
                       height=6.3, aspect=0.9, legend=False)
    plot.set_axis_labels("Type of Travel", "Count")
    plot.fig.suptitle('Type of Travel vs Satisfaction barplot')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    plt.savefig('./Figures/Type_of_Travel_Satisfaction_barplot.png')

    print("Saved Type_of_Travel_Satisfaction_barplot.png in Analysis/Figures/")


def scatter_and_correlate(d, colour, title, filename):
    # This function creates scatterplots
    # find correlation coefficient and slope

    results = stats.linregress(d['Flight Distance'], d['Arrival Delay in Minutes'])
    print(f"r value of Distance vs Arrival Time = {results.rvalue}")

    # plot and save
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Flight Distance', y='Arrival Delay in Minutes', data=d, color=colour)
    plt.title(title)
    plt.savefig(f'./Figures/{filename}')

    print(f"Saved {filename} in Analysis/Figures/")


def class_histogram(d, field, colour, classname, filename, num_bins=20):
    # this function plots a histogram

    hist_data = d[field]

    plt.figure(figsize=(10, 6))
    sns.histplot(hist_data, kde=True, bins=num_bins, color=colour)
    plt.title(f'{field} distribution of {classname} class')
    plt.savefig(f'./Figures/{filename}')

    print(f"Saved {filename} in Analysis/Figures/")

    # we're returning this because we can use it later for the ANOVA TEST.
    return hist_data


fileA = '../Airline Data/train.csv'
fileB = '../Airline Data/test.csv'

data1 = merge_data(fileA, fileB)

class_analysis(data1)

distance_and_delay(data1)

fileA = '../Preprocessed Data/train.csv'
fileB = '../Preprocessed Data/test.csv'

data2 = merge_data(fileA, fileB)

heatmapping(data=data2, title='Heatmap of all Features', colour='rocket', filename='heatmap_all_features.png')

type_of_travel(data1)
