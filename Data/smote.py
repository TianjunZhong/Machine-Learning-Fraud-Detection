from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
import pandas as pd

X = pd.read_csv("data/x.csv")
y = pd.read_csv("data/y.csv")

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Summarize class distribution
print("Before SMOTE: ", Counter(y_train))

# Apply SMOTE
sm = SMOTE(random_state=42, sampling_strategy=0.05)
X_res, y_res = sm.fit_resample(X_train, y_train)

X_train = X_train.loc[:, ~X_train.columns.str.contains('^Unnamed')]
y_train = y_train.loc[:, ~y_train.columns.str.contains('^Unnamed')]
X_test = X_test.loc[:, ~X_test.columns.str.contains('^Unnamed')]
y_test = y_test.loc[:, ~y_test.columns.str.contains('^Unnamed')]

X_train.to_csv("data/xTrain.csv", index=False)
y_train.to_csv("data/yTrain.csv", index=False)
X_test.to_csv("data/xTest.csv", index=False)
y_test.to_csv("data/yTest.csv", index=False)
X_res.to_csv("data/xTrain_smote.csv", index=False)
y_res.to_csv("data/yTrain_smote.csv", index=False)

bf_counter = 0
af_counter = 0
for elem in y_train["Class"]:
    bf_counter += elem
for elem in y_res["Class"]:
    af_counter += elem

print(bf_counter, af_counter, len(y_train["Class"]), len(y_res["Class"]))

# Summarize the new class distribution
print("After SMOTE: ", Counter(y_res))
