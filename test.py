import gd
from sklearn.linear_model import LinearRegression
import sklearn.datasets as ds
import numpy as np

X_train, y_train = ds.make_regression(n_samples=10000, n_features=10, n_informative=2, bias=50.0, noise=20.0, \
                                      random_state=2015)

model = LinearRegression()
model.fit(X_train, y_train)
mse = gd.square_distance_cost_function(X_train, y_train, model.coef_)

print model.intercept_
print model.coef_
print mse

coef, costs = gd.stochastic_gradient_descent(X_train, y_train, cost_function=gd.square_distance_cost_function, verbose=True)
print coef
