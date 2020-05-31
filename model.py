# Multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('Train.csv')
X_train = dataset1.iloc[:, :-1].values
y_train = dataset1.iloc[:, 5].values
X = X_train
Y = y_train

dataset2 = pd.read_csv('Test.csv')
X_test = dataset2.iloc[:, :].values

# Predicting target based on 'All in' method 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_prediction = regressor.predict(X_test)

# Predicting target based on 'Backward elimination' method 
import statsmodels.api as sm
X_train = np.append( arr = np.ones((1600,1)).astype(int) , values = X_train , axis=1)
X_opt = X_train[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS( y_train ,X_opt).fit()
pV = regressor_OLS.pvalues
print(pV)
# All p values are less than 0.05 hence the team of independent variables for backward elimination
# method will as same as ALL IN method therefore resulting in same predicted values for target 