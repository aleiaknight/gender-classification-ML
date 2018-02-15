from sklearn import tree

#variable to hold the classification
clf = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# male and female labels
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#used to train the data 
clf = clf.fit(X, Y)

#store the result and call the method of the decision tree to classify the
#gender given data
prediction = clf.predict([[190, 70, 43]])

print("The data classifies as a: ")
print(prediction)