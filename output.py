import pandas as pd
from preprocessing import preprocess
from exploratory_data_analysis import visualize_data

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

    # Visualize the data
    visualize_data(cleaned_train_set)
    # visualize_data(cleaned_test_set)
