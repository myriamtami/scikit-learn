#!/usr/bin/python3

import pandas as pd
import numpy as np
import sklearn
from sklearn import tree, ensemble
import matplotlib.pyplot as plt
from scipy.stats import rankdata


print(sklearn.__path__)

if __name__ == '__main__':

    sigmas1 = 0.3
    sigmas2 = 0.3
    sigmas = np.array([sigmas1,sigmas2])

    #random.seed(1989)
    #X1 = np.arange(300)/2 #100 valeurs diagonales ecartees de 0.1 de -5 a 5
    ##avec perturbation de 0.01 il restera au moins 0.08 > 0.05 d ecart

    ##X1 = 1 + np.random.rand(100)
    ##X2 = 1 + np.random.rand(100)

    ##pertubations
    #X1p = X1.copy()

    #X1p = X1p.ravel()#sinon la boucle suivante ne marche pas
    #print((np.size(X1))/2)#50
    #perturbations
    #X1p[::2] += 3*((0.5 - np.random.rand(150))/50)
    #X2p = np.random.permutation(X1p)

    #data non sinusoidales
    #Y1 = np.ones(X1p.shape[0])

    #for i in range(X1p.shape[0]):
    #    if (X1p[i] > 100) and (X2p[i] > 100):
    #        Y1[i] = - X1p[i] - X2p[i]
    #        print(X1p[i], X2p[i])
    #    else:
    #        Y1[i] = X1p[i] + X2p[i]

    #Y cloche gaussienne
    #X1p_G = X1p-75
    #X2p_G = X2p-75
    #Y_G = np.exp((-X1p**2 - X2p**2)/2)/(2*np.pi)

    #Y sinusoidales
    X1 = np.random.rand(10)*3.14 #400 pts de 0 a 3.14
    X1p = X1.copy()
    X1p = X1p.ravel()
    X1p[::2] += 3*((0.5 - np.random.rand(5))/50)#1 pt sur 2 perturbe de 0.01 max *3 donc 0.03
    X2p = np.random.permutation(X1p)
    #Y cloches sinusoidales
    Y = (np.sin(X1p + np.pi/4))**2 + (np.cos(X2p + np.pi/4))**2 - 1

    X1p = np.array([1.03669699 ,2.99125051 ,1.70260887 ,1.484833  , 2.95546704 ,0.31776816, 1.79362045, 3.00781948 ,2.91879523 ,2.88190021])
    X2p = np.array([1.03669699 ,2.88190021 ,3.00781948 ,1.79362045 ,1.70260887 ,2.91879523 ,1.484833   ,2.95546704 ,2.99125051 ,0.31776816])
    Y = np.array([ 0.,          0.10008619  ,0.0018919   ,0.3010619  ,-0.05156565  ,0.51230196, -0.3010619 ,  0.04967375 ,-0.06741043 ,-0.54497772])


    print(X1p)
    print(X2p)
    print(Y)



    #Plot verif
    plt.figure()
    plt.scatter(X1p, X2p, s=20, edgecolor="black",c="darkorange", label="data for train")
    plt.xlabel("X1p grille")
    plt.ylabel("X2p grille")
    plt.title("Input variables simulated such as a grid")
    plt.legend()
    plt.show(False)

    #Concatenation
    X1p = X1p.reshape(X1p.shape[0],1)
    X2p = X2p.reshape(X2p.shape[0],1)
    Xp = np.concatenate((X1p, X2p), axis=1)
    #Grille ok
    #je n'ai pas perturbe les Y

    #Allure de Yg a predire
    X_plot = np.arange(X1p.shape[0])+1 #vecteur de 1 a longueur de X1 no obs.



    plt.figure()
    plt.scatter(X_plot, Y, s=20, edgecolor="red", c="darkorange", label="Y des datas simu AVANT perturbe")
    #plt.scatter(X_plot, Yp, s=20, edgecolor="yellow", c="green", label="Y des datas simu perturbe de +-0.01")
    plt.xlabel("rang des obs")
    plt.ylabel("Y")
    plt.title("Y et Yp")
    plt.legend()
    plt.show(False)
    print(np.shape(Xp))

    #REGRESSIONS
    max_depth = 1
    #max_depth = 5
    sigmas=sigmas.ravel()

    # Fit
    regr_Tm = tree.DecisionTreeRegressor(criterion='mse', max_depth=max_depth)
    regr_Tmp = tree.DecisionTreeRegressor(criterion='mseprob',
                                        max_depth=max_depth,
                                        tol=sigmas)
    #regreesions sur le data set complet
    #regr_Tm.fit(Xp, Y)
    #regr_Tmp.fit(Xp, Y)
    # Predict sur train
    #Normal mse prediction
    #y_Tm = regr_Tm.predict(Xp)

    # "Baseline" prediction wiht gaussian regions
    #y_Tm_baseline = regr_Tm.predict2(Xp)

    # Dynamic gaussian regions
    #y_Tmp = regr_Tmp.predict2(Xp)
   #Scores:
    #Returns the coefficient of determination R^2 of the prediction.
    #print(regr_Tm.score(Xp, Y))#
    #print(regr_Tmp.score(Xp, Y))#

    # plot
    #plt.figure()
    #plt.scatter(Y_test, Y_test, s=20, color="black", label="dataTestToPredict", linewidth=2)

    #plt.scatter(Y_test, y_Tm, s=20, color="cornflowerblue",label="mse classic", linewidth=2)
    #plt.scatter(Y_test, y_Tm_baseline, s=20, color="yellowgreen", label="probtree(baseline)", linewidth=2)
    #plt.scatter(Y_test, y_Tmp, s=20, color="yellow", label="probtree(dynamic)", linewidth=2, linestyle='--')

    #plt.xlabel("Y_test")
    #plt.ylabel("predictions")
    #plt.title("Decision Tree Regression 2 input variables simulated")
    #plt.legend()
    #plt.savefig('RF_MseProb_DataSimuGrille2var.png')
    #plt.show()

    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(Xp, test_size = 0.2, random_state=42)
    Xp_train = train_set
    Xp_test = test_set
    #pour Y:
    Y_train_set, Y_test_set = train_test_split(Y, test_size = 0.2, random_state=42)
    Y_train = Y_train_set
    Y_test = Y_test_set

    #regression sur train :
    regr_Tm.fit(Xp_train, Y_train)
    regr_Tmp.fit(Xp_train, Y_train)

    # Predict sur test
    #Normal mse prediction
    y_Tm = regr_Tm.predict(Xp_test)

    # "Baseline" prediction wiht gaussian regions
    y_Tm_baseline = regr_Tm.predict2(Xp_test)

    # Dynamic gaussian regions
    y_Tmp = regr_Tmp.predict2(Xp_test)
   #Scores:
    #Returns the coefficient of determination R^2 of the prediction.
    print(regr_Tm.score(Xp_test, Y_test))#
    print(regr_Tmp.score(Xp_test, Y_test))#

    #Plot arbre
    ############
    print(sigmas)
    from sklearn.tree import export_graphviz
    tree.export_graphviz(regr_Tm, out_file='Tm3_sinPi_newC_newG.dot', rounded=True, filled=True)
    tree.export_graphviz(regr_Tmp, out_file='Tmp3_sinPi_newC_newG.dot', rounded=True, filled=True)
    #Pour convertir le dot : tapper dans le terminal
    #dot -Tpng -O Tm_tree.dot

    # plot
    plt.figure()
    plt.scatter(Y_test, Y_test, s=20, color="black", label="dataTestToPredict", linewidth=2)

    plt.scatter(Y_test, y_Tm, s=20, color="cornflowerblue",label="mse classic", linewidth=2)
    plt.scatter(Y_test, y_Tm_baseline, s=20, color="yellowgreen", label="probtree(baseline)", linewidth=2)
    plt.scatter(Y_test, y_Tmp, s=20, color="yellow", label="probtree(dynamic)", linewidth=2, linestyle='--')

    plt.xlabel("Y_test")
    plt.ylabel("predictions for Y_test")
    plt.title("Decision Tree 3 Regression 2 input variables simulated: sinusoidal case")
    plt.legend()
    plt.savefig('DecisionT3_sinPi_MseProb_DataSimuGrille2var.png')
    #plt.show()
    print(Xp_train)
    print(Y_train)
#CORBEILLE

#remove to close X1 and X2 values
#remove if distance < 0.05
#for i in range(X1p.shape[0]):
#    obs = X1p[i]
#    for ii in range(X1p.shape[0]-1):
#        if abs(obs - X1p[ii]) < 0.05


