import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV


def main():
    X = pd.read_csv("x.csv")
    y = pd.read_csv("y.csv")
    # Applying KNN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # knn = KNeighborsClassifier()
    # param_grid = {'n_neighbors': [1, 3, 5, 7, 10, 15]}

    # grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='roc_auc')
    # grid_search.fit(X_train, y_train)

    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    # print(best_params, best_score)

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    train_aucroc = roc_auc_score(y_train, knn.predict_proba(X_train)[:, 1])
    test_aucroc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
    print(train_aucroc,test_aucroc) #0.9996789553714515 0.9519419136006353

if __name__ == "__main__":
    main()