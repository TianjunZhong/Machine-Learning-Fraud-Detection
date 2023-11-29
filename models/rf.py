from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pandas as pd

# Generating a synthetic dataset for demonstration
X = pd.read_csv("data/x.csv").to_numpy()
y = pd.read_csv("data/y.csv").to_numpy().flatten()

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Creating a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fitting the model
rf.fit(X_train, y_train)

# Making predictions
y_pred = rf.predict(X_test)

# Evaluating the model
report = f1_score(y_test, y_pred)

print(report)
