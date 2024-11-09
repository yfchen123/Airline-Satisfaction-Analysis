import pandas as pd
from preprocessing import preprocess

train_set = pd.read_csv("./Airline Data/train.csv")


if __name__ == "__main__":
    preprocess(train_set)
