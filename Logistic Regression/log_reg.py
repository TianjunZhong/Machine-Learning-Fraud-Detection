from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    # x = pd.read_csv("../Data/x.csv")
    # y = pd.read_csv("../Data/y.csv")
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    x_train = pd.read_csv("../Data/xTrain_smote.csv")
    y_train = pd.read_csv("../Data/yTrain_smote.csv")

    logreg = LogisticRegression()

    class_weights = [{0: i * 0.05, 1: 1 - i * 0.05} for i in range(0, 21)]
    param_grid = {"class_weight": class_weights}
    
    clf = GridSearchCV(logreg, param_grid, cv=5, scoring="f1")
    clf.fit(x_train, y_train)

    # # pd.DataFrame(clf.cv_results_).to_csv("cv_results_class_weight.csv", index=False)
    pd.DataFrame(clf.cv_results_).to_csv("cv_results_class_weight_smote.csv", index=False)
    print(clf.best_params_, clf.best_score_)


    # # logreg = LogisticRegression(class_weight={0: 0.15, 1: 0.85}, solver="saga")
    # logreg = LogisticRegression(class_weight={0: 0.5, 1: 0.5}, solver="saga")

    # param_grid = {"penalty": [None, "l1", "l2"]}

    # clf = GridSearchCV(logreg, param_grid, cv=5, scoring="f1")
    # clf.fit(x_train, y_train)

    # # pd.DataFrame(clf.cv_results_).to_csv("cv_results_penalty.csv", index=False)
    # pd.DataFrame(clf.cv_results_).to_csv("cv_results_penalty_smote.csv", index=False)
    # print(clf.best_params_, clf.best_score_)


if __name__ == "__main__":
    main()