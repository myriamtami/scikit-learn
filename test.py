#!/usr/bin/python3

import sklearn
from sklearn import tree

import numpy as np

print(sklearn.__path__)

# @Warning: don't work with sparse matrix, not implemented for now.
X = [[0, 0], [2, 2], [4,4]]
#Y = [0.5, 2.5, 4.5]
Y = [[0.5,0.6], [2.5,2.3], [4.5, 2.2]]
sigmas = np.random.random(2) * (5 - 0) + 0

print('data X', X)
print('Out Y', Y)


reg = tree.DecisionTreeRegressor(criterion='mse2',
                                 splitter='best2') # why best2 ?!
reg = reg.fit(X, Y)

print(reg)

new = [[0.4, 0.7],
       [2.1,2.2],
       [10,10],
       [0,0]
      ]
res = reg.predict(new)

print('data new', new)
print(res)

