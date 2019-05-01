# TODO: Train the model 
# Attempt "2" -- Just messing around
model2 = DecisionTreeClassifier(max_depth=11, min_samples_leaf=6, min_samples_split=5)
model2.fit(X_train, y_train)

# TODO: Make predictions
y_train_pred_2 = model2.predict(X_train)
y_test_pred_2 = model2.predict(X_test)

# TODO: Calculate the accuracy
train_accuracy_2 = accuracy_score(y_train, y_train_pred_2)
test_accuracy_2 = accuracy_score(y_test, y_test_pred_2)
print('The training accuracy is', train_accuracy_2)
print('The test accuracy is', test_accuracy_2)
