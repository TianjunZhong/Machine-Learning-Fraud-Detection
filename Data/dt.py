import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

def main():
    X = pd.read_csv("x.csv")
    y = pd.read_csv("y.csv")

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Setting up the Decision Tree Classifier
    dt = DecisionTreeClassifier()

    # Defining the parameter grid for Decision Tree
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }

    # Grid Search with Cross-Validation
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    # Train and evaluate the model with best parameters
    dt_best = DecisionTreeClassifier(**best_params)
    dt_best.fit(X_train, y_train)
    train_aucroc = roc_auc_score(y_train, dt_best.predict_proba(X_train)[:, 1])
    test_aucroc = roc_auc_score(y_test, dt_best.predict_proba(X_test)[:, 1])
    print("Train AUC-ROC:", train_aucroc)
    print("Test AUC-ROC:", test_aucroc)

if __name__ == "__main__":
    main()
