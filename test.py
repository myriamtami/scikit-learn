#!/usr/bin/python3

import sklearn
from sklearn import tree

print(sklearn.__path__)

X = [[0, 0], [2, 2], [4,4]]
Y = [0.5, 2.5, 4.5]
reg = tree.DecisionTreeRegressor(criterion="mse2")
reg = reg.fit(X, Y)

print(reg)

res = reg.predict([[1, 1]])
print(res)

