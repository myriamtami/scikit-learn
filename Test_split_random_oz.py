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
    #regr_Tm = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf = min_samples_leaf)
    #print("regr_Tm")
    #regr_Tm_rd = tree.DecisionTreeRegressor(criterion='mse', splitter ="random", min_samples_leaf = min_samples_leaf)
    #print("regr_Tm_random")
    regr_Tmp = tree.DecisionTreeRegressor(criterion='mseprob', min_samples_leaf = min_samples_leaf,tol=sigmas)
    print("regr_Tmp")
    #NEW a tester
    ############
    #regr_Tmp_rd = tree.DecisionTreeRegressor(criterion='mseprob', splitter ="randomprob", min_samples_leaf = min_samples_leaf,tol=sigmas)
    #print("regr_Tmp_random")
    #regr_RFm = ensemble.RandomForestRegressor(criterion='mse', min_samples_leaf = min_samples_leaf,bootstrap=False, n_jobs=1, n_estimators=5)
    #regr_RFmp = ensemble.RandomForestRegressor(criterion='mseprob', min_samples_leaf = min_samples_leaf,tol=sigmas, bootstrap=False,n_jobs=1, n_estimators=5)

    print("Regression Tree with train")
    #regression sur train :
    #regr_Tm.fit(X_train, Y_train)
    #print("regr_Tm.fit")
    #regr_Tm_rd.fit(X_train, Y_train)
    #print("regr_Tm_random.fit")
    #print("Uncertain Regression Tree with train")
    regr_Tmp.fit(X_train, Y_train)
    print("regr_Tmp.fit")
    #regr_Tmp_rd.fit(X_train, Y_train)
    #print("regr_Tmp_random.fit")
    #regr_RFm.fit(X_train, Y_train)
    #regr_RFmp.fit(X_train, Y_train)

    print("Predict sur test ")
    # Predict sur test
    #Normal mse prediction
    #y_Tm = regr_Tm.predict(X_test)
    #y_Tm_rd = regr_Tm_rd.predict(X_test)

    # "Baseline" prediction wiht gaussian regions
    #y_Tm_baseline = regr_Tm.predict2(X_test)

    # Dynamic gaussian regions
    y_Tmp = regr_Tmp.predict2(X_test)
    #y_Tmp_rd = regr_Tmp_rd.predict2(X_test)

    # Forest
    #y_RFm = regr_RFm.predict(X_test)
    #y_RFmp = regr_RFmp.predict(X_test)# use predict2 in intern

    ##On stock les predictions et Y_test
    #conversion en dataframe
    Y_test = pd.DataFrame(Y_test)
    #y_Tm = pd.DataFrame(y_Tm)
    #y_Tm_rd = pd.DataFrame(y_Tm_rd)
    y_Tmp = pd.DataFrame(y_Tmp)
    #y_Tmp_rd = pd.DataFrame(y_Tmp_rd)
    #y_RFm = pd.DataFrame(y_RFm)
    #y_RFmp = pd.DataFrame(y_RFmp)
        
    #Y_test.to_csv('Y_test_ozone_Sig0_1_BM_ss_calib.csv', sep = '\t')
    #y_Tm.to_csv('Y_pred_Tm_ozone_Sig0_1_BM_ss_calib.csv', sep = '\t')
    #y_Tmp.to_csv('Y_pred_Tmp_ozone_Sig0_1_BM_ss_calib.csv', sep = '\t')

    print("sigmas entree ", sigmas)
    #print("y_Tm ", y_Tm)
    print("y_Tmp ", y_Tmp)
    #print("y_Tm_rd ", y_Tm_rd)
    #print("y_Tmp_rd ", y_Tmp_rd)
    print("y_test ", Y_test)
    #RMSE_Tm = np.sqrt(np.mean( (Y_test - y_Tm)**2 ))
    RMSE_Tmp = np.sqrt(np.mean( (Y_test - y_Tmp)**2 ))
    #RMSE_Tm_rd = np.sqrt(np.mean( (Y_test - y_Tm_rd)**2 ))
    #RMSE_Tmp_rd = np.sqrt(np.mean( (Y_test - y_Tmp_rd)**2 ))
    #print("RMSE_Tm ", RMSE_Tm)
    #print("RMSE_Tm_rd ", RMSE_Tm_rd)
    print("RMSE_Tmp ", RMSE_Tmp)
    #print("RMSE_Tmp_rd ", RMSE_Tmp_rd)
    
    #y_RFm.to_csv('Y_pred_RFm_ozone_Sig0_3.csv', sep = '\t')
    #y_RFmp.to_csv('Y_pred_RFmp_ozone_Sig0_3.csv', sep = '\t')
    ####################################################
    ##On lit les fichier data en stock pr les plot en local
    #Y_test = pd.read_csv('Y_test_SinPi400.csv', sep = '\t',index_col=0)

    
   
    
    #Plot arbre
    ############
    #print(sigmas)
    from sklearn.tree import export_graphviz
    #tree.export_graphviz(regr_Tm, out_file='stand_tree_oz.dot', rounded=True, filled=True)
    #tree.export_graphviz(regr_Tm_rd, out_file='stand_tree_rd_oz.dot', rounded=True, filled=True)
    tree.export_graphviz(regr_Tmp, out_file='un_tree_oz.dot', rounded=True, filled=True)
    #tree.export_graphviz(regr_Tmp_rd, out_file='un_tree_rd_oz.dot', rounded=True, filled=True)
    #tree.export_graphviz(regr_Tmp, out_file='Tmp_sinPi_VF400_02_SigZero.dot', rounded=True, filled=True)
    #Pour convertir le dot : tapper dans le terminal
    #dot -Tpng -O stand_tree_oz.dot
    #dot -Tpng -O stand_tree_rd_oz.dot
    #dot -Tpng -O un_tree_oz.dot
    #dot -Tpng -O un_tree_rd_oz.dot

