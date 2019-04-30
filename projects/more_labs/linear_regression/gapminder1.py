# TODO: Add import statements
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
bmi_life_data.head()
life_exp = bmi_life_data[["Life expectancy"]]
bmi = bmi_life_data[["BMI"]]
#features = bmi_life_data.drop("Life expectancy", axis = 1)

#print(life_exp)
#print(bmi)

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(features, bmi, test_size=0.2, random_state=42)
bmi_life_model = LinearRegression()
#bmi_life_model.fit(X_train, y_train)
bmi_life_model.fit(bmi, life_exp)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
