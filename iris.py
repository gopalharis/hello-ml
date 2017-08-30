from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
from sklearn.externals.six import StringIO
import pydot

datasets = load_iris()

test_idx = [0, 50, 100]

train_target = np.delete(datasets.target, test_idx)
train_data = np.delete(datasets.data, test_idx, axis=0)

test_target = datasets.target[test_idx]
test_data = datasets.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
print clf.predict(test_data)

# Graph viz
# conda install -c anaconda pydot=1.0.28
# conda install GraphViz

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=datasets.feature_names,
                     class_names=datasets.target_names,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
