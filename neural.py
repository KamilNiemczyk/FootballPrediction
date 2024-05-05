from prepare_dataset import return_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib


X, y = return_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)


clf = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=500, alpha=0.0001, solver='adam', verbose=10)
clf.fit(X_train, y_train)
# Save the trained model
joblib.dump(clf, 'neural_model.pkl')
print(clf.score(X_test, y_test))



