from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

# Generating a synthetic dataset for demonstration
X = pd.read_csv("data/x.csv").to_numpy()
y = pd.read_csv("data/y.csv").to_numpy().flatten()

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Creating a Multi-layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Fitting the model
mlp.fit(X_train, y_train)

# Making predictions
y_pred_mlp = mlp.predict(X_test)

# Evaluating the model
report_mlp = f1_score(y_test, y_pred_mlp)

print(report_mlp)
