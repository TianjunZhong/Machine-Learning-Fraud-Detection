import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def plot_precision_recall_curve(model, X_test, y_test, label=None):
    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)
    auc_prc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{label} (AUC-PRC = {auc_prc:.2f})')

def main():
    X = pd.read_csv("x.csv")
    y = pd.read_csv("y.csv")

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Setting up the Decision Tree Classifier and Grid Search
    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, criterion='entropy')
    dt.fit(X_train, y_train)
    trainf1 = f1_score(y_train, dt.predict(X_train))
    print(trainf1)
    dt = DecisionTreeClassifier()
    param_grid = {
        'max_depth': [10,None],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # Best parameters and training the model
    best_params = grid_search.best_params_
    best_dt = DecisionTreeClassifier(**best_params)
    best_dt.fit(X_train, y_train)

    # Calculating and printing F1 score for the best model
    best_f1 = f1_score(y_test, best_dt.predict(X_test))
    print(f"Best Parameters: {best_params}")
    print(f"Best Model Test F1 Score: {best_f1:.2f}")

    # Plotting Precision-Recall curves for selected parameter combinations
    plt.figure(figsize=(10, 6))
    for params in grid_search.cv_results_['params'][:10]:  # For example, first 5 combinations
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        label = f"Criteria: {params.get('criterion')}Depth: {params.get('max_depth', 'None')}, Leaf: {params['min_samples_leaf']}"
        plot_precision_recall_curve(model, X_test, y_test, label=label)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Decision Tree')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == "__main__":
    main()
