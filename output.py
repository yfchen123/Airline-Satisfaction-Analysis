import sys
import pandas as pd
from preprocessing import preprocess

sys.path.append('Parameter Tuning')
from classification import classification

train_set = pd.read_csv("./Airline Data/train.csv")
test_set = pd.read_csv("./Airline Data/test.csv")

if __name__ == "__main__":
    # Drop the satisfaction column from both training and testing sets
    X_train = train_set.drop(columns=['satisfaction'])
    y_train = train_set['satisfaction']

    X_test = test_set.drop(columns=['satisfaction'])
    y_test = test_set['satisfaction']

    # Preprocess the set
    cleaned_train_set = preprocess(X_train)
    cleaned_test_set = preprocess(X_test)

    # Package the datasets for classification
    dataset = (cleaned_train_set, y_train, cleaned_test_set, y_test)

    # Perform classification
    classification(dataset)
