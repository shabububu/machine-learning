# $virtualenv venv
# $source venv/bin/activate
# $pip install -U numpy pandas

import pandas as pd

p_data = pd.read_csv("data.csv", header=None)
X = p_data.iloc[:,0:2].values
y = p_data.iloc[:,-1].values

import perceptron
boundary_lines = perceptron.trainPerceptronAlgorithm(X, y)

print(boundary_lines)
# TODO: Add visualization to graph points and boundary lines like graph.png
