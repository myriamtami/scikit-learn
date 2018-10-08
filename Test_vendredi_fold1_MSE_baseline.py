#!/usr/bin/python3

import pandas as pd
import numpy as np
import sklearn
from sklearn import tree, ensemble
import matplotlib.pyplot as plt
from scipy.stats import rankdata

print(sklearn.__path__)

if __name__ == '__main__':
    
    #DATA
    #loading data set ozone after perturbations
    X = pd.read_csv('Xp_ozone_VC_ss_min.csv', sep = '\t',index_col=0)
    #Convert pandas into nd array
    X=X.values

    
    #loading data set splited
    ##fold 1
    Y_fold1 = pd.read_csv('Y_fold1_VC_ozone_ss_minSig.csv', sep = '\t',index_col=0)
    Y_train_fold1 = pd.read_csv('Y_train_fold1_VC_ozone_ss_minSig.csv', sep = '\t',index_col=0)
    X_fold1 = pd.read_csv('X_fold1_VC_ozone_ss_minSig.csv', sep = '\t',index_col=0)
    X_train_fold1 = pd.read_csv('X_train_fold1_VC_ozone_ss_minSig.csv', sep = '\t',index_col=0)
    
    
    ## APRENTISSAGE POUR sigmas_u = 0.05 * sigma_Xp
    ##############################################
    sigma_Xp = np.std(X, axis=0) #calcul des nouveaux sigmas apres perturbation
    print(sigma_Xp)
    #sigmas_u = 0.2 * sigma_Xp
    ###########################
    sigmas_u = 0.05 * sigma_Xp
    #sigmas_u = 0.1 * sigma_Xp
    #sigmas_u = 0.5 * sigma_Xp
    ###########################
    #sigmas_u = sigmas_u.reshape(sigmas_u.shape[0], 1)
    #p = np.shape(X)[1]#10
    #v1 = np.repeat(0.01,p)
    #v1 = v1.reshape(v1.shape[0], 1)
    #v1_v2 = np.concatenate((v1, sigmas_u), axis=1)#concatenation col 1 (avc v1) et col 2 (avc sigmas_u)
    #sigmas_u = np.amin(v1_v2, axis = 1)
    sigmas_u=sigmas_u.ravel()

    print("Apprentissage pour :")
    print("sigmas_u ", sigmas_u)
    print("Et test folds = 1 ")

    X_train_fold1 = X_train_fold1.values
    X_fold1 = X_fold1.values
    Y_train_fold1 = Y_train_fold1.values
    Y_train_fold1 = Y_train_fold1.ravel()
    Y_fold1 = Y_fold1.values
    Y_fold1 = Y_fold1.ravel()
    
    X_train = X_train_fold1
    X_test = X_fold1
    Y_train = Y_train_fold1
    Y_test = Y_fold1

    
    #REGRESSION sur train set
    min_samples_leaf = int(np.shape(X_train)[0]*0.1)#10 pourcent du train
    regr_Tm = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf = min_samples_leaf)
    regr_Tmb = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf = min_samples_leaf,tol=sigmas_u)
    regr_Tm.fit(X_train, Y_train)
    regr_Tmb.fit(X_train, Y_train)
    #PREDICTION sur test_set
    y_Tm = regr_Tm.predict(X_test)
    y_Tmb = regr_Tmb.predict2(X_test)
    #Compute RMSE
    RMSE_Tm1 = np.sqrt(np.mean( (Y_test - y_Tm)**2 ))
    RMSE_Tmb1 = np.sqrt(np.mean( (Y_test - y_Tmb)**2 ))
    print("RMSE_Tm fold 1 ", RMSE_Tm1)
    print("RMSE_Tmb fold 1 ",RMSE_Tmb1)
    ##On stock les predictions et Y_test
    #conversion en dataframe
    Y_test = pd.DataFrame(Y_test)
    y_Tm = pd.DataFrame(y_Tm)
    y_Tmb = pd.DataFrame(y_Tmb)
    print("prediction y MSE ", y_Tm)
    print("prediction y MSE BASELINE ", y_Tmb)
    #Y_test.to_csv('Y_test_fold1_VC_ozone_min0_1Sig.csv', sep = '\t')
    #y_Tm.to_csv('Y_pred_fold1_Tm_VC_ozone_min0_1Sig.csv', sep = '\t')
    #y_Tmp.to_csv('Y_pred_fold1_Tmp_VC_ozone_min0_1Sig.csv', sep = '\t')
