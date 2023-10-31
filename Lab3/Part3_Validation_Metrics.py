import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# from sklearn.datasets import load_boston
# boston = load_boston()

boston = fetch_california_housing()
X = boston.data
Y = boston.target
cv = 10
print("\nlinear regression")
lin = LinearRegression()
scores = cross_val_score(lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))
print("\nridge regression")
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X, Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y, predicted))
print("\nlasso regression")
