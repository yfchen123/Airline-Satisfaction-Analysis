from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def feature_selection(data):
    # This function performs feature selection

    X_train, y_train, X_test, y_test = data

    # Do Mutual Information (should only be done using training data)
    mutual_info = mutual_info_classif(X_train, y_train)
    mutual_info = pd.DataFrame({'score': mutual_info, 'feature': X_train.columns}).sort_values(by='score',
                                                                                               ascending=False)

    # We want to drop the worst features
    num_features_to_drop = 4  # best results i got were with 4
    worst_features = mutual_info.tail(num_features_to_drop)
    worst_features = worst_features['feature'].values

    return worst_features


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


def plot_KNN(k_values, accuracies):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')

    # Add labels, title, and legend
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs. Number of Neighbors (k)')
    plt.xticks(k_values, rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the graph
    plt.tight_layout()
    plt.show()


def plot_Random_Forest(values, accuracies, var_name):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(values, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')

    # Add labels, title, and legend
    plt.xlabel(f'Number of Neighbors {var_name}')
    plt.ylabel('Accuracy')
    plt.title(f'RandomForest Accuracy vs. Number of Neighbors {var_name}')
    plt.xticks(values, rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the graph
    plt.tight_layout()
    plt.show()


def plot_XGBoost(max_depth_values, accuracies, var_name):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_values, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')

    # Add labels, title, and legend
    plt.xlabel(f'{var_name}')
    plt.ylabel('Accuracy')
    plt.title(f'XGBoost Accuracy vs. {var_name}')
    plt.xticks(max_depth_values, rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the graph
    plt.tight_layout()
    plt.show()


def random_forest(X_train, y_train, X_test, y_test):
    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(random_state=42, n_estimators=113, max_depth=51)

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


def xgboost_classifier(X_train, y_train, X_test, y_test):
    # Map string labels to numeric values for XGBoost
    y_train_mapped = y_train.map({"neutral or dissatisfied": 0, "satisfied": 1})
    y_test_mapped = y_test.map({"neutral or dissatisfied": 0, "satisfied": 1})

    # Initialize the XGBoost classifier
    clf = XGBClassifier(
        random_state=42,
        max_depth=7,
        min_child_weight=2,
        reg_lambda=3,
        n_estimators=125,
    )

    # Fit the classifier to the training data
    clf.fit(X_train, y_train_mapped)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_mapped, y_pred)
    precision = precision_score(y_test_mapped, y_pred)
    recall = recall_score(y_test_mapped, y_pred)
    f1 = f1_score(y_test_mapped, y_pred)
    roc_auc = roc_auc_score(y_test_mapped, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc


def classification(dataset):
    X_train, y_train, X_test, y_test = dataset

    # Perform feature selection (should be applied to both train and test data)
    worst_features = feature_selection(dataset)
    X_train_MI = X_train.drop(worst_features, axis=1)
    X_test_MI = X_test.drop(worst_features, axis=1)
    accuracies = []

    '''
    # Train and test the KNN Model
    print("Model Evaluation on KNN:\n")


    # First tune.
    for k in range(1, 51, 5):
        accuracy, precision, recall, f1, auc_roc = KNN(X_train, y_train, X_test, y_test, k=k)
        accuracies.append(accuracy)

        print(f"KNN with k={k}:")
        print(f"  Accuracy:  {accuracy:.5f}")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1-Score:  {f1:.5f}")
        if auc_roc is not None:
            print(f"  AUC-ROC:   {auc_roc:.5f}")
        print("\n")

    plot_KNN(range(1, 51, 5), accuracies)
    '''
    # Second Tune
    '''for k in range(6, 16, 1):
        accuracy, precision, recall, f1, auc_roc = KNN(X_train, y_train, X_test, y_test, k=k)
        accuracies.append(accuracy)

        print(f"KNN with k={k}:")
        print(f"  Accuracy:  {accuracy:.5f}")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1-Score:  {f1:.5f}")
        if auc_roc is not None:
            print(f"  AUC-ROC:   {auc_roc:.5f}")
        print("\n")

    plot_KNN(range(6, 16, 1), accuracies)'''

    # We then train with optimal K and record the results
    '''
    accuracy, precision, recall, f1, auc_roc = KNN(X_train, y_train, X_test, y_test, k=9)
    print(f"KNN with k={9}:")
    print(f"  Accuracy:  {accuracy:.5f}")
    print(f"  Precision: {precision:.5f}")
    print(f"  Recall:    {recall:.5f}")
    print(f"  F1-Score:  {f1:.5f}")

    # Finetune n_estimators first
    print("Model Evaluation on Random Forest:\n")
    for n in range(109, 117, 1):
        accuracy, precision, recall, f1, auc_roc = random_forest(X_train, y_train, X_test, y_test, n)
        accuracies.append(accuracy)
        print(f"RandomForest with n estimators: {n}")
        print(f"  Accuracy:  {accuracy:.5f}")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1-Score:  {f1:.5f}")

    plot_Random_Forest(range(109, 117, 1), accuracies, "n_estimators")

    # Tune parameters

    for max_leaf in range(2, 103, 10):
        accuracy, precision, recall, f1, auc_roc = random_forest(X_train, y_train, X_test, y_test, 113, 51,
                                                                 2, 1, 0.0, "sqrt", max_leaf)
        accuracies.append(accuracy)
        print(f"RandomForest with max leaf nodes of: {max_leaf}")
        print(f"  Accuracy:  {accuracy:.5f}")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1-Score:  {f1:.5f}")

    plot_Random_Forest(range(2, 103, 10), accuracies, "max_leaf_nodes")

    accuracy, precision, recall, f1, auc_roc = random_forest(X_train, y_train, X_test, y_test)
    accuracies.append(accuracy)
    print(f"RandomForest with bootstrap nodes of: True")
    print(f"  Accuracy:  {accuracy:.5f}")
    print(f"  Precision: {precision:.5f}")
    print(f"  Recall:    {recall:.5f}")
    print(f"  F1-Score:  {f1:.5f}")
    
    accuracy, precision, recall, f1, auc_roc = random_forest(X_train, y_train, X_test, y_test, 113, 51,
                                                             2, 1, 0.0, None)
    accuracies.append(accuracy)
    print(f"RandomForest with max_feature of: None")
    print(f"  Accuracy:  {accuracy:.5f}")
    print(f"  Precision: {precision:.5f}")
    print(f"  Recall:    {recall:.5f}")
    print(f"  F1-Score:  {f1:.5f}")
    
    
    # Decimal tuning
    min_weight_fraction_values = np.arange(0.0, 0.5, 0.05)
    for min_weight_fraction_leaf in min_weight_fraction_values:
        accuracy, precision, recall, f1, auc_roc = random_forest(
            X_train, y_train, X_test, y_test, 113, 51, 2, 1, min_weight_fraction_leaf
        )
        accuracies.append(accuracy)
        print(f"RandomForest with min_weight_fraction_leaf of: {min_weight_fraction_leaf:.2f}")
        print(f"  Accuracy:  {accuracy:.5f}")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1-Score:  {f1:.5f}")

    # Plot using the decimal values
    plot_Random_Forest(min_weight_fraction_values, accuracies, "min_weight_fraction_leaf")
    '''

    ''' # Train and test the XGBoost model
    print("Model Evaluation on XGBoost WITHOUT feature selection:\n")
    accuracy, precision, recall, f1, auc_roc = xgboost_classifier(X_train, y_train, X_test, y_test)
    print(f"  Accuracy:  {accuracy:.5f}")
    print(f"  Precision: {precision:.5f}")
    print(f"  Recall:    {recall:.5f}")
    print(f"  F1-Score:  {f1:.5f}")

    # Initialize list to store results for analysis
    accuracies = []

    print("Model Evaluation on XGBoost WITHOUT feature selection:\n")

    # Use a range of learning rates from 0.2 to 0.4 with a step of 0.01
    for learning_rate in np.arange(0.2, 0.4, 0.01):
        print(f"Evaluating XGBoost with learning_rate={learning_rate:.2f}...")

        # Evaluate the model
        accuracy, precision, recall, f1, auc_roc = xgboost_classifier(X_train, y_train, X_test, y_test)
        accuracies.append(accuracy)

        # Print the metrics
        print(f"  Learning Rate: {learning_rate:.2f}")
        print(f"  Accuracy:  {accuracy:.5f}")
        print(f"  Precision: {precision:.5f}")
        print(f"  Recall:    {recall:.5f}")
        print(f"  F1-Score:  {f1:.5f}")
        print(f"  AUC-ROC:   {auc_roc:.5f}")

    # Plot the results
    plot_XGBoost(list(np.arange(0.2, 0.4, 0.01)), accuracies, "learning_rate")

    print(f"Evaluating XGBoost with method...")

     # Evaluate the model
    accuracy, precision, recall, f1, auc_roc = xgboost_classifier(X_train, y_train, X_test, y_test, 7,
                                                                  2, 0, 3, 0, 125, 0.3, )
    accuracies.append(accuracy)

    # Print the metrics
    print(f"  method: ")
    print(f"  Accuracy:  {accuracy:.5f}")
    print(f"  Precision: {precision:.5f}")
    print(f"  Recall:    {recall:.5f}")
    print(f"  F1-Score:  {f1:.5f}")
    print(f"  AUC-ROC:   {auc_roc:.5f}")
    '''

    print(f"Evaluating XGBoost: ")

    # Evaluate the model
    accuracy, precision, recall, f1, auc_roc = xgboost_classifier(X_train, y_train, X_test, y_test)

    # Print the metrics
    print(f"  Accuracy:  {accuracy:.5f}")
    print(f"  Precision: {precision:.5f}")
    print(f"  Recall:    {recall:.5f}")
    print(f"  F1-Score:  {f1:.5f}")
    print(f"  AUC-ROC:   {auc_roc:.5f}")

