from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def KNN(X_train, y_train, X_test, y_test, k=3):
    # Initialize the KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model on the training set
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def classification(dataset):
    X_train, y_train, X_test, y_test = dataset
    k_values = [1, 3, 5, 7, 15, 50]
    for k in k_values:
        accuracy = KNN(X_train, y_train, X_test, y_test, k=k)
        print(f"KNN with k={k}: Accuracy = {accuracy:.2f}")
