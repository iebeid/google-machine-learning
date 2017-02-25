#	Lesson 1 google machine learning
#	Download Anaconda python from https://www.continuum.io/downloads
#	install skikit learn package from http://scikit-learn.org/stable/install.html 

import sklearn
from sklearn import tree

print("hello")

features = [[140, 1],[130, 1],[150, 0],[170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[160, 0]]))