from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

def plot_precision_recall_curve(model, X_test, y_test, label=None):
        probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auc_prc = auc(recall, precision)
        plt.plot(recall, precision, label=f'AUPRC = {auc_prc:.2f})')


def main():
    x = pd.read_csv("../Data/x.csv")
    y = pd.read_csv("../Data/y.csv")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    logreg = LogisticRegression(class_weight={0: 0.15, 1: 0.85}, random_state=42)
    logreg.fit(x_train, y_train)
    y_pred_logreg = logreg.predict(x_test)
    f1 = f1_score(y_test, y_pred_logreg)
    print(f1)

    plot_precision_recall_curve(logreg, x_test, y_test)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Logistic Regression')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()