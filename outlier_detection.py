import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def prepare_data(dataset):

  X_train, y_train, X_test, y_test = dataset

  # create new objects to avoid changing the original objects
  X_train = X_train.copy()
  y_train = y_train.copy()
  X_test = X_test.copy()
  y_test = y_test.copy()

  # change y data to a numerical representation:
  y_train = y_train.map({'satisfied' : 1, 'neutral or dissatisfied' : 0})
  y_test = y_test.map({'satisfied' : 1, 'neutral or dissatisfied' : 0})
  
  # Merge all the data together
  X_train['satisfaction'] = y_train
  X_test['satisfaction'] = y_test
  combined_data = pd.concat([X_train, X_test], ignore_index=True)

  return combined_data

def LOF_detection(pca_data, remove_outliers=False):
  # This function detects outliers and can remove them if desired, via LOF.
  # Also visualizes the outliers in 2 dimensions.

  # LOF needs to be scaled because it relies on distance based metrics.
  scaled_data = StandardScaler().fit_transform(pca_data)
  pca_data.iloc[:, :] = scaled_data

  # detect outliers
  outliers = LocalOutlierFactor(n_neighbors=20, contamination=0.01).fit_predict(pca_data)

  # make a df just for the outliers
  outlier_data = pd.DataFrame(PCA(n_components=2).fit_transform(pca_data), columns=['dimA','dimB'])
  outlier_data['outliers'] = outliers
  
  # make dataframes for outliers and inliers
  outs = outlier_data[outlier_data['outliers'] == -1]
  ins = outlier_data[outlier_data['outliers'] == 1]

  # plot outliers only.
  plt.figure(figsize=(10, 7)) 
  plt.scatter(ins['dimA'], ins['dimB'], color='green', s=30, alpha=0.75)
  plt.scatter(outs['dimA'], outs['dimB'], color='red', s=30, alpha=0.6)
  plt.title(f'LOF outliers')
  plt.xlabel('Dimension 1')
  plt.ylabel('Dimension 2')
  plt.savefig('Analysis/Figures/LOF.png')

  if(remove_outliers == True):
    # If we want, we can update pca_data and remove outliers
    pca_data['outliers'] = outliers
    pca_data = pca_data[pca_data['outliers'] == 1]
    pca_data = pca_data.drop('outliers', axis=1)

  return pca_data

def isolation_forest_detection(data, remove_outliers=False):
  # This function detects outliers via IsolationForest and can remove them if desired.
  # visualizes the outliers.

  # We scale here because we scaled LOF
  scaled_data = StandardScaler().fit_transform(data)
  data.iloc[:, :] = scaled_data
  
  # Do outlier detection on data
  outliers = IsolationForest(contamination=0.01).fit_predict(data)

  # reduce dimensionality to 2 for visualization
  two_dim_data = PCA(n_components=2).fit_transform(data)
  
  # Convert to DataFrame and add outliers for filtering
  two_dim_data = pd.DataFrame(two_dim_data, columns=['dim 1', 'dim 2'])
  two_dim_data['Outliers'] = outliers

  # Filter two dim results into outliers and inliers
  two_dim_outliers = two_dim_data[two_dim_data['Outliers'] == -1]
  two_dim_inliers = two_dim_data[two_dim_data['Outliers'] == 1]

  # Visualize results
  plt.figure(figsize=(10,7))
  plt.scatter(two_dim_inliers['dim 1'], two_dim_inliers['dim 2'], color='g', alpha=0.75)
  plt.scatter(two_dim_outliers['dim 1'], two_dim_outliers['dim 2'], color='r', alpha=0.6)
  plt.title('Outliers vs Inliers following PCA(2)')
  plt.xlabel('Dimension 1')
  plt.ylabel('Dimension 2')
  plt.savefig('Analysis/Figures/isolation_forest_outliers.png')

  if(remove_outliers == True):
    # Remove outliers if desired.
    data['Outliers'] = outliers
    data = data[data['Outliers'] == 1]
    data = data.drop('Outliers', axis=1)

  return data

def detect_outliers(dataset):

  print("if an error message appears here, you can ignore it, it is a bug.")
  # prepare data for outlier detection
  data = prepare_data(dataset)

  # PROBLEM: LOF is computationally demanding for datasets over 10000 records and over 50 features.
  # The dataset has 25 features, but well over 1000000 records. We can sample to reduce by size. This
  # works well for detecting global outliers. But, we want both models to work on the same data. 
  data = data.sample(frac=0.25)

  LOF_detection(data)

  isolation_forest_detection(data)

  return
  