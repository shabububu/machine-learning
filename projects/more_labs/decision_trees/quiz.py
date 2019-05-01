# $ virtualenv venv
# $ source venv/bin/activate
# $ pip install -U scikit-learn numpy pandas

# Overfitted decision tree example from  Udacity Machine Learning Nanodegree

# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
# model = DecisionTreeClassifier() # acc= 1.0
# model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10) # acc = .83
model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 1) # acc = 1.0
#model = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 1) # acc = 0.99

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
print(model.get_params())
acc = accuracy_score(y, y_pred)
print("Accuracy: {}".format(acc))
