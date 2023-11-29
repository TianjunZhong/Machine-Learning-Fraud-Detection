from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    x = pd.read_csv("../Data/x.csv")
    y = pd.read_csv("../Data/y.csv")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    logreg = LogisticRegression()

    class_weights = []
    for i in range(0, 21):
        class_weights.append({0: i * 0.05, 1: 1 - i * 0.05})

    parameters = {"penalty": [None, "l1", "l2"],
                  "class_weight": class_weights}
    
    clf = GridSearchCV(logreg, parameters, cv=5, scoring="f1")
    clf.fit(x_train, y_train)

    pd.DataFrame(clf.cv_results_).to_csv("cv_results.csv", index=False)
    print(clf.best_params_, clf.best_score_)


if __name__ == "__main__":
    main()