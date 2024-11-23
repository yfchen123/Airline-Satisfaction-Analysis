from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def random_forest(X_train, y_train, X_test, y_test):
    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(random_state=42)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="satisfied")
    recall = recall_score(y_test, y_pred, pos_label="satisfied")
    f1 = f1_score(y_test, y_pred, pos_label="satisfied")
    roc_auc = roc_auc_score(y_test.map({"neutral or dissatisfied": 0, "satisfied": 1}), y_pred_proba)

    # Return metrics
    return accuracy, precision, recall, f1, roc_auc


def KNN(X_train, y_train, X_test, y_test, k=3):
    # Initialize the KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model on the training set
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1] if hasattr(knn, "predict_proba") else None

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="satisfied")
    recall = recall_score(y_test, y_pred, pos_label="satisfied")
    f1 = f1_score(y_test, y_pred, pos_label="satisfied")
    auc_roc = roc_auc_score(y_test.map({"neutral or dissatisfied": 0, "satisfied": 1}),
                            y_proba) if y_proba is not None else None

    return accuracy, precision, recall, f1, auc_roc


def classification(dataset):
    X_train, y_train, X_test, y_test = dataset

    # Train and test the KNN Model
    '''k_values = [1, 3, 5, 7, 15, 50]

    print("Model Evaluation on KNN:\n")
    for k in k_values:
        accuracy, precision, recall, f1, auc_roc = KNN(X_train, y_train, X_test, y_test, k=k)
        print(f"KNN with k={k}:")
        print(f"  Accuracy:  {accuracy:.2f}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1-Score:  {f1:.2f}")
        if auc_roc is not None:
            print(f"  AUC-ROC:   {auc_roc:.2f}")
        print("\n")'''

    '''# Train and test the randomForest model
    print("Model Evaluation on Random Forest:\n")
    accuracy, precision, recall, f1, auc_roc = random_forest(X_train, y_train, X_test, y_test)
    print(f"  Accuracy:  {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  F1-Score:  {f1:.2f}")'''


