from sklearn import tree
features = [[140, 0], [130, 0], [150, 1], [170, 1]]
labels = ["apple", "apple", "orange", "orange"]
clf = tree.DecisionTreeClassifier()   # Empty Classifier  
clf =  clf.fit(features, labels)    # Fit - Find Patterns in data 
print clf.predict([160, 1])