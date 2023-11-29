import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def plot_precision_recall_curve(model, X_test, y_test, label=None):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    auc_prc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{label} (AUC-PRC = {auc_prc:.2f})')

def plot_roc_curve(model, X_test, y_test, label=None):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=label)

def main():
    X = pd.read_csv("x.csv")
    y = pd.read_csv("y.csv")

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Setting up KNN and the grid search
    # knn = KNeighborsClassifier()
    # param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 15, 17, 19, 21]}
    # grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='average_precision')
    # grid_search.fit(X_train, y_train)

    # # Best parameters from grid search
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    # print("Best Parameters:", best_params)
    # print("Best Average Precision Score:", best_score)

    # # Training KNN with the best parameters
    best_knn = KNeighborsClassifier(n_neighbors=7)
    best_knn.fit(X_train, y_train)

    # ROC AUC scores
    train_f1 = f1_score(y_train, best_knn.predict(X_train))
    test_f1 = f1_score(y_test, best_knn.predict(X_test))
    print("Train F1 Score:", train_f1)
    print("Test F1 Score:", test_f1)
    # plt.figure(figsize=(10, 8))
    # for n_neighbors in [1, 3, 5, 7, 10, 15, 17, 19, 21]:  # Selecting a subset of n_neighbors for demonstration
    #     model = KNeighborsClassifier(n_neighbors=n_neighbors)
    #     model.fit(X_train, y_train)
    #     plot_precision_recall_curve(model, X_test, y_test, label=f'n_neighbors={n_neighbors}')
    
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curves for KNN')
    # plt.legend(loc='lower left')
    # plt.show()
if __name__ == "__main__":
    main()