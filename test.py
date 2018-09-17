#!/usr/bin/python3

import sklearn
print(sklearn.__path__)
from sklearn import tree

import numpy as np


# @Warning: don't work with sparse matrix, not implemented for now.
X = [[0, 0], [2, 2], [4,4]]
#X = [[0], [2], [4]]
Y = [0.5, 2.5, 4.5]
#Y = [[0.5,0.6], [2.5,2.3], [4.5, 2.2]]
sigmas = 0.000001

print('data X', X)
print('Out Y', Y)


reg = tree.DecisionTreeRegressor(criterion='mseprob',
                                 tol=sigmas)
reg = reg.fit(X, Y)

print(reg)

t = reg.tree_

print('value', t.value)
print('feature', t.feature)
print('n_node_samples', t.n_node_samples)
print('weighted_n_node_samples', t.weighted_n_node_samples)
print('threshold', t.threshold)
print('left_child', t.children_left)
print('right_child', t.children_right)
print('y_test', t.y)

print('Preg', t.preg)


new = [[0.4, 0.7],
       [2.1, 2.2],
       [10,  10],
       [0,   0]
      ]

#new = [[0.4],
#       [2.1],
#       [10],
#       [0],
#      ]


print('data new', new)
print('predict 1:', reg.predict(new))
print('Predict 2:', reg.predict2(new))

