from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
features = [[140, 0], [130, 0], [150, 1], [170, 1]]
labels = ["apple", "apple", "orange", "orange"]
clf = tree.DecisionTreeClassifier()   # Empty Classifier  
clf = clf.fit(features, labels)    # Fit - Find Patterns in data
print clf.predict([160, 1])

# Graph viz
# conda install -c anaconda pydot=1.0.28
# conda install GraphViz

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=["weight", "texture"],
                     class_names=["apple", "orange"],
                     filled=True, rounded=True,
                     impurity=False)


graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("applevsorange.pdf")