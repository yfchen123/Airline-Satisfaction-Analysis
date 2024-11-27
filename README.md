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

Hyperparameter tuning:
So for KNN the rough graph I started with looks like this:
![](Parameter%20Tuning/KNNFirsttune.png)

This suggests that the peak is around k = 11, hence I will tune it to be from 6-16, to find the max K.
Next is the graph for zooming in to those 6-16 which further suggests that a k of 9 is optimal so I
will be using the k = 9 for this KNN model:
![](Parameter%20Tuning/KNNFineTuned.png)

The results for KNN with optimal parameter:

Model Evaluation on KNN:

KNN with k=9:
  Accuracy:  0.93244
  Precision: 0.94658
  Recall:    0.89669
  F1-Score:  0.92096

So for RandomForest the first parameter to tune is n_estimators, I will do a fairly broad search first:
Model Evaluation on Random Forest:

RandomForest with n estimators: 100
  Accuracy:  0.96324
  Precision: 0.97310
  Recall:    0.94230
  F1-Score:  0.95745
RandomForest with n estimators: 200
  Accuracy:  0.96285
  Precision: 0.97248
  Recall:    0.94203
  F1-Score:  0.95701
RandomForest with n estimators: 300
  Accuracy:  0.96285
  Precision: 0.97239
  Recall:    0.94212
  F1-Score:  0.95702
RandomForest with n estimators: 400
  Accuracy:  0.96289
  Precision: 0.97240
  Recall:    0.94221
  F1-Score:  0.95706
RandomForest with n estimators: 500
  Accuracy:  0.96277
  Precision: 0.97230
  Recall:    0.94203
  F1-Score:  0.95693
RandomForest with n estimators: 600
  Accuracy:  0.96293
  Precision: 0.97214
  Recall:    0.94256
  F1-Score:  0.95712
RandomForest with n estimators: 700
  Accuracy:  0.96289
  Precision: 0.97205
  Recall:    0.94256
  F1-Score:  0.95708
RandomForest with n estimators: 800
  Accuracy:  0.96277
  Precision: 0.97205
  Recall:    0.94230
  F1-Score:  0.95694
RandomForest with n estimators: 900
  Accuracy:  0.96281
  Precision: 0.97231
  Recall:    0.94212
  F1-Score:  0.95697

The values seem to suggest that the value of 100 is best for this application: I will tune it for values
around 100, probably 1-200.
![](Parameter%20Tuning/RandomForestN_estimators_tune.png)

It is really hard to see from this graph but the optimal is around 81-121 so I will tune for those 
only:
![img.png](Parameter%20Tuning/RandomForestN_estimators_2.png)

Okay I will try to not repeat so many graphs for all the parameters, they will all be in the parameter
tuning directory instead I will show the final graphs for each parameter:

N_estimators final graph:
![](Parameter%20Tuning/RandomForestN_estimators_final.png)

113 is optimal here is the results:

RandomForest with n estimators: 113
  Accuracy:  0.96347
  Precision: 0.97320
  Recall:    0.94273
  F1-Score:  0.95773

Gini:
RandomForest with n estimators: 113
  Accuracy:  0.96347
  Precision: 0.97320
  Recall:    0.94273
  F1-Score:  0.95773

Entropy:
RandomForest with n estimators: 113
  Accuracy:  0.96196
  Precision: 0.97216
  Recall:    0.94028
  F1-Score:  0.95596

log_loss:
RandomForest with n estimators: 113
  Accuracy:  0.96196
  Precision: 0.97216
  Recall:    0.94028
  F1-Score:  0.95596

As a result it seems Gini is superior.

Next final depth graph:
No improvements after 51 so:

![](Parameter%20Tuning/RandomForest_depth.png)
We will be using 51 max-depth.








