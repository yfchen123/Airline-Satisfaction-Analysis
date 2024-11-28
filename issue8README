RandomForest without any Parameter%20Tuning gets us great results right out of the box:

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
need much hyperParameter%20Tuning, as well it seems hyperParameter%20Tuning does not help KNN much.

HyperParameter%20Tuning:
So for KNN the rough graph I started with looks like this:
![](Parameter%20Tuning/KNN/KNNFirsttune.png)

This suggests that the peak is around k = 11, hence I will tune it to be from 6-16, to find the max K.
Next is the graph for zooming in to those 6-16 which further suggests that a k of 9 is optimal so I
will be using the k = 9 for this KNN model:
![](Parameter%20Tuning/KNN/KNNFineTuned.png)

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
![](Parameter%20Tuning/RF/RFN_estimators_tune.png)

It is really hard to see from this graph but the optimal is around 81-121 so I will tune for those 
only:
![img.png](Parameter%20Tuning/RF/RFN_estimators_2.png)

Okay I will try to not repeat so many graphs for all the parameters, they will all be in the parameter
tuning directory instead I will show the final graphs for each parameter:

N_estimators final graph:
![](Parameter%20Tuning/RF/RFN_estimators_final.png)

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

![](Parameter%20Tuning/RF/RF_depth.png)
We will be using 51 max-depth.

It seems smaller min_samples is better, so we will just use the minimum which is 2:
![](Parameter%20Tuning/RF/RF_min_samples.png)

The same is true for min_samples_leaf, so we will be using the default:
![](Parameter%20Tuning/RF/RF.png)

As before lower is better for min_weight_fraction:
![](Parameter%20Tuning/RF/RF_min_weight.png)

Next for max_features:

Sqrt:
RandomForest with max_feature of: sqrt
  Accuracy:  0.96347
  Precision: 0.97320
  Recall:    0.94273
  F1-Score:  0.95773

log2:
RandomForest with max_feature of: log2
  Accuracy:  0.96347
  Precision: 0.97320
  Recall:    0.94273
  F1-Score:  0.95773

None:
RandomForest with max_feature of: None
  Accuracy:  0.96185
  Precision: 0.96986
  Recall:    0.94238
  F1-Score:  0.95592

Well it looks like just leaving it as the default is good.

The optimal parameters for RandomForest is n_estimators = 113 and max_depth of 51.

Result:
  Accuracy:  0.96347
  Precision: 0.97320
  Recall:    0.94273
  F1-Score:  0.95773


XGBoost:
Feature selection seems to lower results:
Model Evaluation on XGBoost WITHOUT feature selection:

  Accuracy:  0.96354
  Precision: 0.97082
  Recall:    0.94537
  F1-Score:  0.95792

Model Evaluation on XGBoost WITH feature selection (Mutual Information):

  Accuracy:  0.96297
  Precision: 0.97019
  Recall:    0.94466
  F1-Score:  0.95726

Hence, I will not be doing it.

XGBoost for max depth:
![](Parameter%20Tuning/XGBoost/XGBoost_max_depth_final.png)

7 seems to be optimal:
Evaluating XGBoost with max_depth=7...
  max_depth: 7
  Accuracy:  0.96408
  Precision: 0.97222
  Recall:    0.94519
  F1-Score:  0.95851
  AUC-ROC:   0.99510

Here is the graph for min_child_weight:
![](Parameter%20Tuning/XGBoost/XGBoost_min_child_weight.png)

2 seems to be optimal:
Evaluating XGBoost with min_child_weight=2...
  min_child_weight: 2
  Accuracy:  0.96416
  Precision: 0.97146
  Recall:    0.94615
  F1-Score:  0.95864
  AUC-ROC:   0.99504

Gamma attribute:
![](Parameter%20Tuning/XGBoost/XGBoost_gamma.png)

Seems like the default of 0 is best:
Evaluating XGBoost with gamma=0...
  gamma: 0
  Accuracy:  0.96416
  Precision: 0.97146
  Recall:    0.94615
  F1-Score:  0.95864
  AUC-ROC:   0.99504

Next doing lambda:
![](Parameter%20Tuning/XGBoost/XGBoost_lambda.png)

The optimal appears to be 3:
Evaluating XGBoost with reg_lambda=3...
  reg_lambda: 3
  Accuracy:  0.96424
  Precision: 0.97197
  Recall:    0.94580
  F1-Score:  0.95871
  AUC-ROC:   0.99524

Alpha:
Leaving it as 0 seems good:

Evaluating XGBoost with alpha=0...
  alpha: 0
  Accuracy:  0.96424
  Precision: 0.97197
  Recall:    0.94580
  F1-Score:  0.95871
  AUC-ROC:   0.99524

n_estimators:
125 and 126 have the same accuracy, so I will just use 125.
![](Parameter%20Tuning/XGBoost/XGBoost_n_estimators_final.png)

Results for 125:

Evaluating XGBoost with n_estimators=125...
  n_estimators: 125
  Accuracy:  0.96431
  Precision: 0.97155
  Recall:    0.94642
  F1-Score:  0.95882
  AUC-ROC:   0.99519

Graph for Learning Rate:
![](Parameter%20Tuning/XGBoost/Learning_rate_final.png)

Optimal seems to be 0.3:
Evaluating XGBoost with learning_rate=0.30...
  Learning Rate: 0.30
  Accuracy:  0.96431
  Precision: 0.97155
  Recall:    0.94642
  F1-Score:  0.95882
  AUC-ROC:   0.99519

GPU_Hist:
  Accuracy:  0.96312
  Precision: 0.97088
  Recall:    0.94431
  F1-Score:  0.95741
  AUC-ROC:   0.99510

Hist:
  Accuracy:  0.96431
  Precision: 0.97155
  Recall:    0.94642
  F1-Score:  0.95882
  AUC-ROC:   0.99519

Approx:
  Accuracy:  0.96347
  Precision: 0.97082
  Recall:    0.94519
  F1-Score:  0.95783
  AUC-ROC:   0.99506

Exact:
  Accuracy:  0.96374
  Precision: 0.97083
  Recall:    0.94580
  F1-Score:  0.95816
  AUC-ROC:   0.99528

I will just stick with auto since it gives the best performance.






