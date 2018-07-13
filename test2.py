#!/usr/bin/python3

import numpy as np
import sklearn
from sklearn import tree, ensemble
import matplotlib.pyplot as plt

print(sklearn.__path__)

if __name__ == '__main__':

    # Gen data
    #sigmas = 0.00000001
    sigmas = 0.5
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    #X = np.asarray((X[:,0],X[:,0]+1)).T
    y[::5] += 3 * (0.5 - np.random.rand(16))

    print(X.shape)
    print(y.shape)

    max_depth = 5
    #max_depth = 100

    # Fit
    regr_3 = ensemble.RandomForestRegressor(criterion='mseprob',
                                            max_depth=max_depth,
                                            tol=sigmas, bootstrap=False)
    regr_1 = tree.DecisionTreeRegressor(criterion='mse',
                                        max_depth=max_depth,
                                        tol=sigmas)
    regr_2 = tree.DecisionTreeRegressor(criterion='mseprob',
                                        max_depth=max_depth,
                                        tol=sigmas)


    regr_3.fit(X, y)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.05)[:, np.newaxis]
    #X_test = np.asarray((X_test[:,0],X_test[:,0]+1)).T

    # Normal mse prediction
    y_1 = regr_1.predict(X_test)

    # "Baseline" prediction wiht gaussian regions
    y_2 = regr_1.predict2(X_test)

    # Dynamic gaussian regions
    y_3 = regr_2.predict2(X_test)

    # Forest
    y_4 = regr_3.predict(X_test) # use predict2 in intern

    # plot
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black",
                c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue",
             label="mse", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="probtree(baseline)", linewidth=2)
    plt.plot(X_test, y_3, color="yellow", label="probtree(dynamic)", linewidth=2, linestyle='--')
    plt.plot(X_test, y_4, color="blue", label="probtree(dynamic+forest)", linewidth=2, linestyle='--')

    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()
