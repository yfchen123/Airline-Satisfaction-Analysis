# Airline-Satisfaction-Analysis
Analysis on Airline Satisfaction

You should clone main branch for the new issues and create a new branch.  

PLEASE DO NOT MERGE INCORRECT OR NOT WORKING CODE ONTO THE MAIN BRANCH, DO NOT MERGE BRANCHES WITHOUT A PULL REQUEST.

Issue 3: Note that the dataset is very skewed for some attributes so this may present an issue when doing
classification or clustering. Perhaps you should check these before doing decision trees or rely on some
technique that does not assume a normal distribution. 

Classification: For the classification I started with KNN, I noticed that the accuracy is very similar across
different values of K and starts to drop for large K. Suggesting that we do not need a large number of neighbor
points to make an accurate prediction. 

The table looks like the following:

KNN with k=1:
  Accuracy:  0.92
  Precision: 0.91
  Recall:    0.90
  F1-Score:  0.90
  AUC-ROC:   0.91

KNN with k=3:
  Accuracy:  0.93
  Precision: 0.94
  Recall:    0.90
  F1-Score:  0.92
  AUC-ROC:   0.96

KNN with k=5:
  Accuracy:  0.93
  Precision: 0.95
  Recall:    0.90
  F1-Score:  0.92
  AUC-ROC:   0.98

KNN with k=7:
  Accuracy:  0.93
  Precision: 0.95
  Recall:    0.90
  F1-Score:  0.92
  AUC-ROC:   0.98

KNN with k=15:
  Accuracy:  0.93
  Precision: 0.95
  Recall:    0.89
  F1-Score:  0.92
  AUC-ROC:   0.98

KNN with k=50:
  Accuracy:  0.92
  Precision: 0.95
  Recall:    0.87
  F1-Score:  0.91
  AUC-ROC:   0.98

RandomForest without any parameter tuning gets us great results right out of the box:

Model Evaluation on Random Forest:

  Accuracy:  0.96
  Precision: 0.97
  Recall:    0.94
  F1-Score:  0.96

The accuracy is much higher than possible with KNN. Suggesting that Random Forest is an excellent choice 
for this problem. 

I tried to use SVM however due to the vastness of my dataset, the algorithm will not terminate, so I decided
to switch to a more efficient algorithm called XGBoost.

XGBoost gives me a similar result out of the box as well so it seems to be good for this dataset:
Model Evaluation on XGBoost:

  Accuracy:  0.96
  Precision: 0.97
  Recall:    0.95
  F1-Score:  0.96

In general, it appears that for my dataset XGBoost and Random Forest work well out of the box and do not 
need much hyperparameter tuning, as well it seems hyperparameter tuning does not help KNN much.


