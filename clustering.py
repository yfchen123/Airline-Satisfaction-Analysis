import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from kmodes.kprototypes import KPrototypes

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

def kmeans_dbscan_clusters(data):
  # This function returns a 2d visualization of some clustering (and also applies outlier detection if desired)

  outlier_message = ''
  outlier_file_modifier = ''

  pca_data = data.copy()
  pca_data = pca_data.dropna()

  # sample data to improve runtimes since there's a lot of data.
  pca_data = pca_data.sample(frac=0.1)
  
  # Scale
  scaled_data = StandardScaler().fit_transform(pca_data)
  pca_data.iloc[:,:] = scaled_data
  
  # Drop stuff that might affect clustering (alot of these are highly correlated)
  pca_data = pca_data.drop([
                            'Inflight wifi service',
                            'Departure/Arrival time convenient',
                            'Ease of Online booking',
                            'Gate location',
                            #'Food and drink',
                            'Online boarding',
                            'Seat comfort',
                            'Inflight entertainment',       # drop whatever you want from these.
                            'On-board service',             # comment a feature out to keep it in the clustering calc.
                            'Leg room service',
                            'Baggage handling',
                            'Checkin service',
                            'Inflight service',
                            'Cleanliness',
                            #'Departure Delay in Minutes',   # always leave out one of departure/arrival delay (extreme correlation)
                            'Arrival Delay in Minutes',
                            #'Customer Type',
                            #'Type of Travel',
                            'Class_Business', 'Class_Eco', 'Class_Eco Plus'
                            ], axis=1)


  # get clustering
  num_kmeans_clusters = 5
  clustering_kmeans = KMeans(n_clusters=num_kmeans_clusters, init='k-means++').fit_predict(pca_data)
  clustering_dbscan = DBSCAN(eps=2.1, min_samples=26).fit_predict(pca_data)

  # silhouette_scores
  score = silhouette_score(pca_data, clustering_kmeans)
  print(f'KMeans Silhouette Score: {score:.3f}')
  score = silhouette_score(pca_data, clustering_dbscan)
  print(f'DBSCAN Silhouette Score: {score:.3f}')

  # reduce dimensionality of data for 2D plotting
  pca_data = pd.DataFrame(PCA(2).fit_transform(pca_data), columns=['dimA','dimB'])
  pca_data['kmeans'] = clustering_kmeans
  pca_data['dbscan'] = clustering_dbscan
  
  # Plot clusters and save
  # Kmeans
  plt.figure(figsize=(10, 7))
  plt.scatter(pca_data['dimA'], pca_data['dimB'], c=pca_data['kmeans'], cmap='viridis', s=30, alpha=0.5)
  plt.title(f'Kmeans clustering with {num_kmeans_clusters} clusters')
  plt.xlabel('PC 1')
  plt.ylabel('PC 2')
  plt.savefig(f'Analysis/Figures/kmeans.png')

  # DBSCAN
  plt.figure(figsize=(10, 7))
  plt.scatter(pca_data['dimA'], pca_data['dimB'], c=pca_data['dbscan'], cmap='viridis', s=30, alpha=0.5)
  plt.title(f'dbscan clustering')
  plt.xlabel('PC 1')
  plt.ylabel('PC 2')
  plt.savefig(f'Analysis/Figures/dbscan.png')

def k_prototypes_cluster(data):

  print('a')
  pca_data = data.copy()

  # Very computationally heavy method. Made data size a lot smaller for testing. Change it however you please.
  pca_data = pca_data.sample(frac=0.01)

  # Get columns for categorical and numerical data
  categorical_data = ['Gender', 'Customer Type', 'Type of Travel', 'satisfaction', 'Class_Business','Class_Eco','Class_Eco Plus']
  numerical_data = list(set(pca_data.columns) - set(categorical_data))

  # Scale numerical data
  scaled_data = MinMaxScaler().fit_transform(pca_data[numerical_data])
  pca_data[numerical_data] = scaled_data
  
  # get indeces for categorical data
  categorical_index = []
  for c in categorical_data:
    categorical_index.append(pca_data.columns.get_loc(c))

  # cluster using k prototypes
  clustering = KPrototypes(n_clusters=5).fit_predict(pca_data, categorical=categorical_index)

  #Get silhouette score.
  score = silhouette_score(pca_data, clustering)
  print(f'K_prototypes Silhouette Score: {score:.3f}')
  
  # Reduce to two dimensions for plotting
  pca_data = pd.DataFrame(PCA(2).fit_transform(pca_data), columns=['dimA','dimB'])
  pca_data['cluster'] = clustering

  # Plot clusters and save
  plt.figure(figsize=(10, 7))
  plt.scatter(pca_data['dimA'], pca_data['dimB'], c=pca_data['cluster'], cmap='viridis', s=30, alpha=0.5)
  plt.title(f'Kmeans clustering with 5 clusters')
  plt.xlabel('PC 1')
  plt.ylabel('PC 2')
  plt.savefig(f'Analysis/Figures/k_prototypes.png')

def cluster(dataset): 

  # prepare data for outlier detection
  data = prepare_data(dataset)

  # k_prototypes_cluster(data)
  kmeans_dbscan_clusters(data)

  k_prototypes_cluster(data)

