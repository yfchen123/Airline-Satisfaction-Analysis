import pandas as pd
from preprocessing import preprocess

train_set = pd.read_csv("./Airline Data/train.csv")
test_set = pd.read_csv("./Airline Data/test.csv")


if __name__ == "__main__":
    # Drop the satisfaction column from both training and testing sets
    X_train = train_set.drop(columns=['satisfaction'])
    y_train = train_set['satisfaction']

    X_test = test_set.drop(columns=['satisfaction'])
    y_test = test_set['satisfaction']

    # Preprocess the set
    preprocess(train_set)
    preprocess(test_set)
