from sklearn import tree
features = [[140, 0], [130, 0], [150, 1], [170,1], [180,0]]
labels = ["apple", "apple", "orange", "orange", "apple"]
clf = tree.DecisionTreeClassifier()
clf =  clf.fit(features, labels)
print clf.predict([179, 0])

from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, 
                    out_file=dot_data, 
                    feature_names=features,
                    class_names=labels,
                    filled=True, rounded=True,
                    impurity=False)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydot.graph_from_dot_data(dot_data)
graph.write_pdf("gopal.pdf")