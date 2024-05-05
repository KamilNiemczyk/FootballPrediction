from prepare_dataset import return_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = return_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
# DecisionTreeClassifier
# tree = DecisionTreeClassifier()   # DecisionTreeClassifier
# tree.fit(X_train, y_train)
# print("DecisionTreeClassifier score")
# print(tree.score(X_test, y_test))

# plt.figure(figsize=(20,10))  # Set the figure size
# plot_tree(tree, filled=True, feature_names=['player_id','situation', 'X', 'Y','shotType', 'h_team'], max_depth=8)
# plt.show()
# r = export_text(tree, feature_names=['player_id','situation', 'X', 'Y','shotType', 'h_team','a_team'])
# print(r)

        
# k-NN, k=3
# neighbours = KNeighborsClassifier(n_neighbors=3)
# neighbours_fit = neighbours.fit(X_train, y_train)
# predict = neighbours_fit.predict(X_test)
# correct = accuracy_score(y_test, predict)
# print("k-NN, k=3")
# print(correct)
# print(confusion_matrix(y_test, predict))

# k-NN, k=5
# neighbours = KNeighborsClassifier(n_neighbors=5)
# neighbours_fit = neighbours.fit(X_train, y_train)
# predict = neighbours_fit.predict(X_test)
# correct = accuracy_score(y_test, predict)
# print("k-NN, k=5")
# print(correct)
# print(confusion_matrix(y_test, predict))

# Naive Bayes
gnb = GaussianNB()
gnb_fit = gnb.fit(X_train, y_train)
predict = gnb_fit.predict(X_test)
correct = accuracy_score(y_test, predict)
print("Naive Bayes")
print(correct)




    


