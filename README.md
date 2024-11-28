# CMPT 454 – Airline Survey Analysis Project
Fall 2024\
Yun Fei Chen (\<student number\>)\
David Krljanovic (301427415)

## Table of Contents

- [Introduction](#introduction)

- [Preprocessing](#preprocessing)

- [Exploratory Data Analysis](#exploratory-data-analysis)

- [Clustering](#clustering)

- [Classification](#classification)

- [Conclusions](#conclusions)

## Introduction
Spirit Airlines Inc. is a popular American airline known for it's affordable airfare, and poor service and overall customer experience. Frequent delays, uncomfortable seats, and questionably terrible in-flight service plagues travelers on their journies to their target destinations. One could argue that the sole purpose of a plane is to transport people from point A to point B, and that selling an experience is not a necessity. This begs the question: Does the quality of these services actually influence the satisfaction of airline customers to a significant degree? This is a good question to ask – a business certainly does not wish to deter its target demographics from considering them for future travels. It is worth noting that as of recent, Spirit Airlines stock has fallen ninety-five percent in the the last year, and the company has filed for bankruptcy. 

Using a [Kaggle-sourced dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data) containing over 100,000 airline customer satisfaction survey results of 24 features, we will perform data mining techniques to uncover hidden patterns and information that may prove to be useful to data scientists and businesses alike. Following a routine preprocessing of the data, exploratory data analysis will reveal surface level information that will serve to guide the procedure of machine learning clustering and classification tasks. Each section will serve as an overview of our procedure, and the final section will conclude our findings and attempt to present a meaningful discovery.

## Preprocessing

## Exploratory Data Analysis

Before running data mining tasks such as clustering and classification, it is important that we have some basic understandings concerning the distribution of our dataset. We can determine simple relationships and trends using basic inferential statistics and visualization techniques. We also decided to take a closer look at the relationships between specific key features that may help explain the results of future analysis in the clustering and classification tasks.

### Airline Classes:

Our first instinct was that the airline classes would have a disparity between reported satisfaction, and we fully expected to see business class have a significantly higher count of satisfied customers over the economy and economy plus classes. Our instincts were correct: when we plotted a seaborn bar plot of the count of satisfied and dissatisfied/neutral customers for each class, the business class was not only the class with the highest count of satisfied customers, but it was also the only class with the majority of customers satisfied. 

<img src="Analysis/Figures/class_satisfaction_barplot.png" alt="Project Diagram" width="400"/>

Our analysis on classes did not end here, however. When we plotted the age distribution of customers across each class using a seaborn histogram, we found that the means of the distributions appeared to be different:

<img src="Analysis/Figures/economy_age_dist.png" alt="Project Diagram" width="400"/><img src="Analysis/Figures/economy_plus_age_dist.png" alt="Project Diagram" width="400"/><img src="Analysis/Figures/business_age_dist.png" alt="Project Diagram" width="400"/>

Applying the ANOVA test to the different age distributions of each class, we found that there is most definitely a significant difference of the means between the different airline classes (p-value of 1.05e-228). Following this result, we ran post-hoc analysis using a scipy Tukey Test, which indeed determined that all the age distributions had mean ages that significantly differed from one another.

From both the class satisfaction bar plot and the age distribution histogram results, we inferred the following:

- Airline class strongly influences the satisfaction of customers. When running classification algorithms, we have a strong predictor of satisfaction.

- The mean customer age of each airline class is significantly different; since airline class and age are related, and class and satisfaction are related, we can say that the two features are associated, and age is a somewhat reliable predictor of satisfaction. 

### Flight Distances:

We also sought a relationship between continuous features. Two features that stood out to us were the 'Flight Distance' and 'Arrival Delay in Minutes' feature; we were expecting that on longer flights, the room for estimation error by the airline grew larger, which could contribute to arrival delays. However, this was not the case. Before plotting the features against each other, we removed all records that had a departure delay greater than 0 minutes. This is done to prevent comparing arrival delays with flight distances where the departure delay is likely influencing arrival delay the most. Finally, plotting 'Flight Distance' against 'Arrival Delay in Minutes', we find no evidence of strong linear correlation, with a correlation coefficient of r = 0.0268. 

<img src="Analysis/Figures/distance_delay_scatterplot.png" alt="Project Diagram" width="400"/>

Perhaps we were not looking deep enough – we decided to break down the data further by dividing data into satisfied and dissatisfied/neutral groups. Again, we found no evidence of linear correlation (r = 0.0422 and 0.059). There does not appear to be any direct linear relationship between arrival delay and flight distances within these contexts. 

As a follow-up, we were curious whether the Flight Distance, a predetermined statistic for each customer, would influence the results of satisfaction among customers. What we were surprised to find was that the mean flight distance for satisfied customers happened to be over 600 kilometers greater than non-satisfied customers. 

<img src="Analysis/Figures/distance_distribution_satisfied.png" alt="Project Diagram" width="400"/>

Could this be a sign that airlines are aware that longer flights are more taxing on travellers, and so they provide a higher standard of service than on shorter flights?

### Heatmapping:

Instead of checking each pair of features for any signs of correlation manually, we were able to do it all at once using a correlation matrix, enabled by the data preprocessing that handled categorical features. By using the pandas DataFrame corr() function on our preprocessed data and the seaborn library, we were able to discover more interesting correlational relationships in our dataset:

<img src="Analysis/Figures/heatmap_all_features.png" alt="Project Diagram" width="800"/>

What immediately stood out is a moderate positive correlation between Flight Distance and customers flying in business class, and a moderate negative correlation between Flight Distance and customers flying in economy class. Knowing what we learned using barplots to plot satisfaction against airline class, now our discovery that satisfied customers have a longer average flight distance is much more clear. If business class flyers are more likely to be on longer flights, and economy class flyers are less likely to be on longer flights, then the proportion of satisfied customers with longer flight times will likely be much higher among satisfied customers, since satisfied customers are typically business class flyers. 

Very similar results are evident in the type of travel; customers flying for business are more likely to travel with business class. We wondered if we plotted the satisfaction of business vs personal travellers, whether we would find similar results to the plot of airline class vs satisfaction, and we did. Business travelers, of which business class flyers make up a bigger proportion than Personal travelers, report higher satisfaction levels than economy flyers, and the majority of business travelers report satisfaction:

<img src="Analysis/Figures/Type_of_Travel_Satisfaction_barplot.png" alt="Project Diagram" width="400"/>


The heatmap also identifies a consistent correlation disparity between the different elements of a flight experience and airline classes as well. Business class typically positively correlates with a greater customer satisfaction in each service, and economy class typically negatively correlates. Moreover, the different categories of in-flight experience influence each other as well. For example, a higher satisfaction in flight cleanliness positively correlates moderately with satisfaction in the food a drink category.

## Outlier Detection

## Clustering

## Feature selection

## Classification

## Conclusions