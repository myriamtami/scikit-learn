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
    ozone = pd.read_csv('ozone.csv', sep=';', decimal = ',', header = 0, index_col=0,
                        dtype = {'vent' : str, 'pluie' : str})
    X = ozone.drop(['maxO3', 'vent', 'pluie'], axis = 1)#suppression target et categorical variables
    Y = ozone['maxO3']
    #X.dtypes #Nature des variables
       
    #Convert pandas into nd array
    X=X.values
    Y=Y.values
    Y=Y.ravel()
    

    
    p = np.shape(X)[1]
    n = np.shape(X)[0]
   
    #division train/test du data set
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size = 0.2, random_state=42)
    #pour Y:
    Y_train, Y_test = train_test_split(Y, test_size = 0.2, random_state=42)

    
    
    #REGRESSIONS
    min_samples_leaf = int(np.shape(X_train)[0]*0.1) #10 pourcent des observations du train c'est bien
    #SIGMAS
    BM = np.std(X, axis=0)
    sigmas_tol_0 = np.repeat(0.000000001,p)
    #sigmas = np.repeat(0.3,p)
    sigmas = sigmas_tol_0
    #sigmas = sigmas_tol_1
    #sigmas = sigmas_tol_2
    #sigmas = BM
    
    sigmas=sigmas.ravel()

    # Fit
 
    regr_Tmp_rd = tree.DecisionTreeRegressor(criterion='mseprob', splitter ="randomprob", min_samples_leaf = min_samples_leaf,tol=sigmas)
    print("regr_Tmp_random")
    regr_RFm = ensemble.RandomForestRegressor(criterion='mse', min_samples_leaf = min_samples_leaf,bootstrap=False, n_jobs=1, n_estimators=100)
    print("regr_RFm")
    regr_RFm_rd = ensemble.RandomForestRegressor(criterion='mse',  splitter ="randomprob", min_samples_leaf = min_samples_leaf,bootstrap=False, n_jobs=1, n_estimators=100)
    print("regr_RFm_rd")
    #regr_RFmp = ensemble.RandomForestRegressor(criterion='mseprob', min_samples_leaf = min_samples_leaf,tol=sigmas, bootstrap=False,n_jobs=1, n_estimators=5)
    #print("regr_RFmp")
    #NEW a tester
    ############
    regr_RFmp_rd = ensemble.RandomForestRegressor(criterion='mseprob', splitter ="randomprob", min_samples_leaf = min_samples_leaf,tol=sigmas, bootstrap=False,n_jobs=1, n_estimators=100)
    print("regr_RFmp_rd")
    
    print("Regression Tree with train")
    #regression sur train :
    #print("Uncertain Regression Tree with train")
    #regr_Tmp.fit(X_train, Y_train)
    #print("regr_Tmp.fit")
    regr_Tmp_rd.fit(X_train, Y_train)
    print("regr_Tmp_random.fit")
    regr_RFm.fit(X_train, Y_train)
    print("regr_RFm.fit")
    regr_RFm_rd.fit(X_train, Y_train)
    print("regr_RFm_rd.fit")
    #regr_RFmp.fit(X_train, Y_train)
    #print("regr_RFmp.fit")
    regr_RFmp_rd.fit(X_train, Y_train)
    print("regr_RFmp_rd.fit")

    print("Predict sur test ")
    # Predict sur test
    # Dynamic gaussian regions
    y_Tmp_rd = regr_Tmp_rd.predict2(X_test)
    # Forest
    y_RFm = regr_RFm.predict(X_test)
    y_RFm_rd = regr_RFm_rd.predict(X_test)
    #y_RFmp = regr_RFmp.predict(X_test)# use predict2 in intern
    y_RFmp_rd = regr_RFmp_rd.predict(X_test)# use predict2 in intern

    ##On stock les predictions et Y_test
    #conversion en dataframe
    Y_test = pd.DataFrame(Y_test)
    y_Tmp_rd = pd.DataFrame(y_Tmp_rd)
    y_RFm = pd.DataFrame(y_RFm)
    y_RFm_rd = pd.DataFrame(y_RFm_rd)
    #y_RFmp = pd.DataFrame(y_RFmp)
    y_RFmp_rd = pd.DataFrame(y_RFmp_rd)
        
    #Y_test.to_csv('Y_test_ozone_Sig0_1_BM_ss_calib.csv', sep = '\t')
    #y_Tm.to_csv('Y_pred_Tm_ozone_Sig0_1_BM_ss_calib.csv', sep = '\t')
    #y_Tmp.to_csv('Y_pred_Tmp_ozone_Sig0_1_BM_ss_calib.csv', sep = '\t')

    print("sigmas entree ", sigmas)
    #print("y_Tmp_rd", y_Tmp_rd)
    #print("y_RFm ", y_RFm)
    #print("y_RFmp ", y_RFmp)
    #print("y_RFmp_rd ", y_RFmp_rd)
    #print("y_test ", Y_test)
    RMSE_Tmp_rd = np.sqrt(np.mean( (Y_test - y_Tmp_rd)**2 ))
    RMSE_RFm = np.sqrt(np.mean( (Y_test - y_RFm)**2 ))
    RMSE_RFm_rd = np.sqrt(np.mean( (Y_test - y_RFm_rd)**2 ))
    #RMSE_RFmp = np.sqrt(np.mean( (Y_test - y_RFmp)**2 ))
    RMSE_RFmp_rd = np.sqrt(np.mean( (Y_test - y_RFmp_rd)**2 ))
    print("RMSE_Tmp_rd ", RMSE_Tmp_rd)
    print("RMSE_RFm ", RMSE_RFm)
    print("RMSE_RFm_rd ", RMSE_RFm_rd)
    #print("RMSE_RFmp ", RMSE_RFmp)
    print("RMSE_RFmp_rd ", RMSE_RFmp_rd)
    
    #y_RFm.to_csv('Y_pred_RFm_ozone_Sig0_3.csv', sep = '\t')
    #y_RFmp.to_csv('Y_pred_RFmp_ozone_Sig0_3.csv', sep = '\t')
    ####################################################
    ##On lit les fichier data en stock pr les plot en local
    #Y_test = pd.read_csv('Y_test_SinPi400.csv', sep = '\t',index_col=0)

    
   

