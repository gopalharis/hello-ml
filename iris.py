from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

datasets = load_iris()

test_idx = [0, 50, 100]

train_target = np.delete(datasets.target, test_idx)
train_data = np.delete(datasets.data, test_idx, axis=0)

test_target = datasets.target[test_idx]
test_data = datasets.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
print clf.predict(test_data)

