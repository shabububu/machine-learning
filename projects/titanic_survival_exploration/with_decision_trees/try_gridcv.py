# Attempt "3" -- Use GridSearchCV
# NOTE: Cutting and pasting from previous project (Boston Housing Prices)

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    #Calculate the performance score between 'y_true' and 'y_predict'
    score = accuracy_score(y_true, y_predict)
    
    # Return the score
    return score

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    #Create a decision tree classifier object
    classifier = DecisionTreeClassifier()

    # Create a dictionary for the parameters
    #params = {'max_depth': range(1,21), 'min_samples_leaf':range(1,11)}
    params = {'max_depth': range(1,21), 'min_samples_leaf':range(1,11), 'min_samples_split':[2, 4, 6, 8, 10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    # NOTE: Helpful link let me know that cv=cv_sets was necessary -- https://gist.github.com/mariogintili/7d98f1f3efc3bc4cd4e0dea160bcdb72
    grid = GridSearchCV(classifier, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
print("Parameter 'min_samples_leaf' is {} for the optimal model.".format(reg.get_params()['min_samples_leaf']))
print("Parameter 'min_samples_split' is {} for the optimal model.".format(reg.get_params()['min_samples_split']))

# Train the model with optimal settings
# Attempt "4" -- Just messing around
model4 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, min_samples_split=5)
model4.fit(X_train, y_train)

# TODO: Make predictions
y_train_pred_4 = model4.predict(X_train)
y_test_pred_4 = model4.predict(X_test)

# TODO: Calculate the accuracy
train_accuracy_4 = accuracy_score(y_train, y_train_pred_4)
test_accuracy_4 = accuracy_score(y_test, y_test_pred_4)
print('The training accuracy is', train_accuracy_4)
print('The test accuracy is', test_accuracy_4)
