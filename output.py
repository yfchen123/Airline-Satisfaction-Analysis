import sys
import pandas as pd
from preprocessing import preprocess

sys.path.append('Parameter Tuning')
from classification import classification
from outlier_detection import detect_outliers
from clustering import cluster
from exploratory_data_analysis import doEDA

train_set = pd.read_csv("./Airline Data/train.csv")
test_set = pd.read_csv("./Airline Data/test.csv")

if __name__ == "__main__":
    # Drop the satisfaction column from both training and testing sets
    X_train = train_set.drop(columns=['satisfaction'])
    y_train = train_set['satisfaction']

    X_test = test_set.drop(columns=['satisfaction'])
    y_test = test_set['satisfaction']

    print('\n PROPROCESSING \n')
    # Preprocess the set
    cleaned_train_set = preprocess(X_train)
    cleaned_test_set = preprocess(X_test)

    # Package the datasets for classification
    dataset = (cleaned_train_set, y_train, cleaned_test_set, y_test)

    print('\n EDA \n')
    doEDA()

    print('\n OUTLIER DETECTION \n')
    # Perform Outlier Detection
    detect_outliers(dataset)
    
    print('\n CLUTSERING \n ')
    # Perform Clustering
    cluster(dataset)
    
    print('\n CLASSIFICATION \n ')
    # Perform Feature selection and Classification
    classification(dataset)

