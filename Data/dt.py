import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
def plot_roc_curve(model, X_test, y_test, label=None):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

def main():
    X = pd.read_csv("x.csv")
    y = pd.read_csv("y.csv")

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Setting up the Decision Tree Classifier
    dt = DecisionTreeClassifier()

    # Defining the parameter grid for Decision Tree
    param_grid = {
        'max_depth': [None, 10],
        'min_samples_leaf': [1, 8],
        'criterion': ['gini', 'entropy']
    }

    # Grid Search with Cross-Validation
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
    # grid_search.fit(X_train, y_train)

    # # Best parameters and score
    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    # print("Best Parameters:", best_params)
    # print("Best Score:", best_score)



    # Train and evaluate the model with best parameters
    dt_best = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=8)
    dt_best.fit(X_train, y_train)
    train_aucroc = roc_auc_score(y_train, dt_best.predict_proba(X_train)[:, 1])
    test_aucroc = roc_auc_score(y_test, dt_best.predict_proba(X_test)[:, 1])
    print("Train AUC-ROC:", train_aucroc)
    print("Test AUC-ROC:", test_aucroc) 
    #Best Parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 8}
    # Best Score: 0.9210252246537729
    # Train AUC-ROC: 0.9957812660910474
    # Test AUC-ROC: 0.9232914994218115

   
    # grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
    # grid_search.fit(X_train, y_train)
    # plt.figure(figsize=(10, 8))
    # for params in grid_search.cv_results_['params'][:10]:  # Selecting first 5 combinations for demonstration
    #     model = DecisionTreeClassifier(**params)
    #     model.fit(X_train, y_train)
    #     plot_roc_curve(model, X_test, y_test, label=f'{params}')
    
    # plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='No Skill')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curves')
    # plt.legend(loc='lower right')
    # plt.show()

if __name__ == "__main__":
    main()
