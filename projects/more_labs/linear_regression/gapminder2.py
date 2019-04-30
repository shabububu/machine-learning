# Mods to run in my local environment rather than Udacity's:
# $virtualenv venv
# $source venv/bin/activate
# $pip install -U scikit-learn
# $pip install -U numpy pandas

# TODO: Add import statements
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assign the dataframe to this variable.
# TODO: Load the data
#bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
bmi_life_data = pd.read_csv("/FULL/PATH/HERE/bmi_life_expectancy.csv")
bmi_life_data.head()
life_exp = bmi_life_data[["Life expectancy"]]
bmi = bmi_life_data[["BMI"]]

print(life_exp)
print(bmi)

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi, life_exp)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])

print(laos_life_exp)
