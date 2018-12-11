#!/usr/bin/python3

import os.path
import sys, getopt

import multiprocessing as mp

import pandas as pd
import numpy as np
import sklearn
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import rankdata

print(sklearn.__path__)

# Activer multithreading
mt = True

# Fonction multithreadé
def multithreading_computation(queue, method_tune, X_sub_train, Y_sub_train, X_valid, Y_valid, sigmas_u, j, eta_j, min_samples_leaf_percent, max_depth, RMSE, t):

    sigmas_temp = sigmas_u.copy()
    sigmas_temp[j] = sigmas_temp[j] + t * eta_j

    #print("[LOG] * Thread ", t, " | sigmas_temp", sigmas_temp)
    for elem in range(0, len(sigmas_temp)):
        print("[LOG] * Process ", t, " | sigmas_temp[",elem,"]", sigmas_temp[elem])

    #REGRESSION sur sub_train set
    min_samples_leaf = 1
    if min_samples_leaf_percent != 1: #A rentrer si on veut le parametre par defaut
        min_samples_leaf = int(np.shape(X_sub_train)[0]*min_samples_leaf_percent)#10 pourcent des observations du train c'est bien max_depth_parameter

    if method_tune == "Tmseprob":
        regr_Tmseprob = tree.DecisionTreeRegressor(criterion='mseprob', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_temp)
        regr_Tmseprob.fit(X_sub_train, Y_sub_train)
        #PREDICTION sur valid_set
        y_Tmseprob = regr_Tmseprob.predict2(X_valid)
        RMSE_test = np.sqrt(np.mean( (Y_valid - y_Tmseprob)**2 ))

    elif method_tune == "TmsePredprob":
        regr_TmsePredprob = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_temp)
        regr_TmsePredprob.fit(X_sub_train, Y_sub_train)
        #PREDICTION sur valid_set
        y_TmsePredprob = regr_TmsePredprob.predict2(X_valid)
        RMSE_test = np.sqrt(np.mean( (Y_valid - y_TmsePredprob)**2 ))

    print("[LOG] * Process ", t, " | RMSE old ", RMSE, " RMSE_test ", RMSE_test, " sigmas_u[", j, " + ", t * eta_j,"] ", sigmas_temp[j])

    res = [RMSE_test, sigmas_temp[j]]
    queue.put(res)

# Fonction Gridsearch
def cvgridsearch(method, method_tune, X, X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, passe, step, min_samples_leaf_percent, max_depth):
    #INITIALISATION
    p = np.shape(X)[1]
    n = np.shape(X)[0]
    sigmas_u = np.repeat(0.00000000000000000000001,p)#traiter le cas null
    sigma_Xp = np.std(X, axis=0)#calcul des nouveaux sigmas apres perturbation
    M = sigma_Xp.copy()
    m = sigmas_u.copy()
    eta = M/step
    RMSE = 10000000000000

    sigmas_u=sigmas_u.ravel()

    print("[LOG] Learning for sigmas_u ")
    for elem in range(0, len(sigmas_u)):
        print("[LOG] * sigmas_u[",elem,"]", sigmas_u[elem])

    Y_train_fold = Y_train_fold.ravel()
    Y_test_fold = Y_test_fold.ravel()

    X_train = X_train_fold
    X_test = X_test_fold
    Y_train = Y_train_fold
    Y_test = Y_test_fold

    #division sub_train/validation du train set
    X_sub_train, X_valid = train_test_split(X_train, test_size = 0.2, random_state=42)
    #pour Y:
    Y_sub_train, Y_valid = train_test_split(Y_train, test_size = 0.2, random_state=42)

    if passe == 0:
        n_count = 2*p#nb de passes choisi
    else :
        n_count = passe

    i_count = 1#indice des passes
    while i_count < n_count + 1:
        print("[LOG] Starting pass number ", i_count, " on ", n_count)
        for j in range(p):
            T = (M[j] - m[j])/eta[j]#Au cas ou interv differ suivant j ds grille
            sigmas_u[j] = 0.00000000000000000000001#pour ne pas demarrer au dernier sigmas_u[j] calibré pour j (et depasser la grille) mais garder les sigmas_u[j'] calibres pour les j' differents de j
            print("[LOG] ****")
            print("[LOG] * Starting computation for  j ", j, " | pass ", i_count)

            if mt:
                processes = []
                queue = mp.Queue()
                results_RMSE = [{} for t in range(int(T)+1)]
                results_Sigmas_u = [{} for t in range(int(T)+1)]
                results = []

                for elem in range(0, len(sigmas_u)):
                    print("[LOG] * sigmas_u[",elem,"]", sigmas_u[elem])

                for t in range(int(T)+1):
                    process = mp.Process(target=multithreading_computation, args=[queue, method_tune, X_sub_train, Y_sub_train, X_valid, Y_valid, sigmas_u, j, eta[j], min_samples_leaf_percent, max_depth, RMSE, t])
                    processes.append(process)
                    process.start()

                for process in processes:
                    res = queue.get() # will block
                    results.append(res)

                for process in processes:
                    process.join()

                for t in range(int(T)+1):
                    res = results[t]
                    results_RMSE[t] = res[0]
                    results_Sigmas_u[t] = res[1]

                results_RMSE = np.asarray(results_RMSE)
                results_Sigmas_u = np.asarray(results_Sigmas_u)
                min_index_RMSE_test = np.argmin(results_RMSE)

                print("[LOG] * End of computation for j ", j, " | RMSE old ", RMSE, " RMSE_test_min ", results_RMSE[min_index_RMSE_test], "sigmas_u[j] opt ", results_Sigmas_u[min_index_RMSE_test])
                print("[LOG] ****")

                sigmas_u[j] = results_Sigmas_u[min_index_RMSE_test]
                RMSE = results_RMSE[min_index_RMSE_test]

            else:
                for t in range(int(T)+1):
                    print("[LOG] * sigmas_u[j] ", sigmas_u[j])
                    print("[LOG] * sigmas_u ")
                    for elem in range(0, len(sigmas_u)):
                        print("[LOG] * sigmas_u[",elem,"]", sigmas_u[elem])
                    #REGRESSION sur sub_train set
                    min_samples_leaf = 1
                    if min_samples_leaf_percent != 1: #A rentrer si on veut le parametre par defaut
                        min_samples_leaf = int(np.shape(X_sub_train)[0]*min_samples_leaf_percent)#10 pourcent des observations du train c'est bien max_depth_parameter

                    if method_tune == "Tmseprob":
                        regr_Tmseprob = tree.DecisionTreeRegressor(criterion='mseprob', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_u)
                        regr_Tmseprob.fit(X_sub_train, Y_sub_train)
                        #PREDICTION sur valid_set
                        y_Tmseprob = regr_Tmseprob.predict2(X_valid)
                        RMSE_test = np.sqrt(np.mean( (Y_valid - y_Tmseprob)**2 ))

                    elif method_tune == "TmsePredprob":
                        regr_TmsePredprob = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_u)
                        regr_TmsePredprob.fit(X_sub_train, Y_sub_train)
                        #PREDICTION sur valid_set
                        y_TmsePredprob = regr_TmsePredprob.predict2(X_valid)
                        RMSE_test = np.sqrt(np.mean( (Y_valid - y_TmsePredprob)**2 ))

                    print("[LOG] * RMSE old ", RMSE, " RMSE_test ", RMSE_test, " sigmas_u[j] ", sigmas_u[j])

                    if (RMSE_test <= RMSE):
                        sigmas_u_opt = sigmas_u[j]
                        RMSE = RMSE_test

                    sigmas_u[j] = sigmas_u[j] + eta[j]#LE CAS ZERO EST BIEN PRIS EN COMPTE ICI
                    sigmas_u=sigmas_u.ravel()
                print("[LOG] * End of computation for j ", j, " | RMSE old ", RMSE, " RMSE_test ", RMSE_test, "sigmas_u[j] opt ", sigmas_u_opt)
                print("[LOG] ****")
                sigmas_u[j] = sigmas_u_opt#garde en memoire mais apres on ajoute eta et on depasse M : il faut re initialiser le sigmas_u a 1.e-23

        #Une passe effectuee
        sigmas_u=sigmas_u.ravel()#le dernier sigmas est garde en memoire et non le sigmas de depart
        #on repart a la passe suivante avec un premier sigmas_u calibre
        print("[LOG] Ending pass number ", i_count, " on ", n_count)
        print("[LOG] sigmas calibrated")
        for elem in range(0, len(sigmas_u)):
            print("[LOG] sigmas_u[",elem,"]", sigmas_u[elem])
        #passe suivante
        i_count = i_count + 1

    sigmas_u=sigmas_u.ravel()#le dernier sigmas est garde en memoire et non le sigmas de depart
    print("[LOG] sigmas calibrated after ", i_count - 1, " passes")
    for elem in range(0, len(sigmas_u)):
        print("[LOG] sigmas_u[",elem,"]", sigmas_u[elem])
    #APPRENDRE ARBRE SUR TRAIN COMPLET avec sigmas_u calibre
    #REGRESSION sur train set
    min_samples_leaf = 1
    if min_samples_leaf_percent != 1: #A rentrer si on veut le parametre par defaut
        min_samples_leaf = int(np.shape(X_train)[0]*min_samples_leaf_percent)#10 pourcent du train

    print(sigmas_u)
    if method == "Tmse":
        regr_Tmse = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        regr_Tmse.fit(X_train, Y_train)
        #PREDICTION sur test_set
        y_pred = regr_Tmse.predict(X_test)
    elif method == "Tmseprob":
        regr_Tmseprob = tree.DecisionTreeRegressor(criterion='mseprob', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_u)
        regr_Tmseprob.fit(X_train, Y_train)
        #PREDICTION sur test_set
        y_pred = regr_Tmseprob.predict2(X_test)
    elif method == "TmsePredprob":
        regr_TmsePredprob = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_u)
        regr_TmsePredprob.fit(X_train, Y_train)
        #PREDICTION sur test_set
        y_pred = regr_TmsePredprob.predict2(X_test)

    #Compute RMSE
    RMSE = np.sqrt(np.mean( (Y_test - y_pred)**2 ))
    print("[LOG] RMSE fold ", RMSE)

    sigmas_u = pd.DataFrame(sigmas_u)
    sigmas_u.to_csv('sigmas_u_cv.csv', sep='\t', encoding='utf-8')

# Fonction Cross-validation
def crossvalidation(sigmas_u, method, X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, min_samples_leaf_percent, max_depth):

    print("[LOG] learning for sigmas_u ")
    for elem in range(0, len(sigmas_u)):
        print("[LOG] sigmas_u[",elem,"]", sigmas_u[elem])

    Y_train_fold = Y_train_fold.ravel()
    Y_test_fold = Y_test_fold.ravel()

    X_train = X_train_fold
    X_test = X_test_fold
    Y_train = Y_train_fold
    Y_test = Y_test_fold

    #REGRESSION sur train set
    min_samples_leaf = 1
    if min_samples_leaf_percent != 1: #A rentrer si on veut le parametre par defaut
        min_samples_leaf = int(np.shape(X_train)[0]*min_samples_leaf_percent)#10 pourcent du train

    print(sigmas_u)
    if method == "Tmse":
        regr_Tmse = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        regr_Tmse.fit(X_train, Y_train)
        #PREDICTION sur test_set
        y_pred = regr_Tmse.predict(X_test)
    elif method == "Tmseprob":
        regr_Tmseprob = tree.DecisionTreeRegressor(criterion='mseprob', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_u)
        regr_Tmseprob.fit(X_train, Y_train)
        #PREDICTION sur test_set
        y_pred = regr_Tmseprob.predict2(X_test)
    elif method == "TmsePredprob":
        regr_TmsePredprob = tree.DecisionTreeRegressor(criterion='mse', min_samples_leaf=min_samples_leaf, max_depth=max_depth, tol=sigmas_u)
        regr_TmsePredprob.fit(X_train, Y_train)
        #PREDICTION sur test_set
        y_pred = regr_TmsePredprob.predict2(X_test)

    #Compute RMSE
    RMSE = np.sqrt(np.mean( (Y_test - y_pred)**2 ))
    print("[LOG] RMSE fold ", RMSE)

# Main uncertain trees
if __name__ == '__main__':
    inputfile = ''
    outputfile = ''
    sigmas_ufile = 'EmptyFile'
    fold = 0
    method = ''
    method_tune = ''
    action = ''
    passe = 0
    step = 0
    min_samples_leaf_percent = 0.0
    max_depth = 0

    #Arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:u:f:m:t:a:p:s:l:d:",["ifile=","ofile=","sigmas_ufile=","fold=","method=","methodtune=","action=", "passe=", "step=","minsampleleafpercent=","maxdepth="])
    except getopt.GetoptError:
        print('uncertain_trees.py -i <inputfile> -o <outputfile> -u <sigmas_ufile> -f <fold> -m <method> -t <methodtune> -a <action> -p <passe> -s <step> -l <minsampleleafpercent> -d <maxdepth>')
        sys.exit(2)
    for opt, arg in opts:
        print("[LOG] -----------")
        print("[LOG] ", opt , " " , arg)
        if opt == '-h':
            print('---- Gridsearch ')
            print('uncertain_trees.py -i <inputfile> -o <outputfile> -f <fold> -m <method> -t <methodtune> -a cvgridsearch -p <passe> -s <step> -l <minsampleleafpercent> -d <maxdepth>')
            print('-i <inputfile> Matrix of inputs variables')
            print('-o <outputfile> Vector of target variable ')
            print('-f <fold> The fold index')
            print('-m <method> Method to apply among standard regression tree (Tmse), uncertain regression tree (Tmseprob) or standard regression tree with prediction rule handling uncertain inputs (TmsePredprob)')
            print('      Tmse | Tmseprob | TmsePredprob')
            print('-t <methodtune> Method to use during the grid search to tune the parameter sigmas_u')
            print('      Tmseprob | TmsePredprob')
            print('-a <action> Action to do among a grid search with cross-validation to tune the parameter of uncertainty (sigma_u) or a cross validation without tuning')
            print('      cvgridsearch | cv')
            print('-p <passe> ')
            print('-s <step> ')
            print('-l <minsampleleafpercent> ')
            print('-d <maxdepth> ')
            print('')
            print('---- Cross validation ')
            print('uncertain_trees.py -i <inputfile> -o <outputfile> -f <fold> -m <method> -a cv -l <minsampleleafpercent> -d <maxdepth>')
            print('-i <inputfile> Matrix of inputs variables')
            print('-o <outputfile> Vector of target variable ')
            print('-u <sigmas_ufile> Vector sigmas_u ')
            print('-f <fold> The fold index')
            print('-m <method> Method to apply among standard regression tree (Tmse), uncertain regression tree (Tmseprob) or standard regression tree with prediction rule handling uncertain inputs (TmsePredprob)')
            print('      Tmse | Tmseprob | TmsePredprob')
            print('-a <action> Action to do among a grid search with cross-validation to tune the parameter of uncertainty (sigma_u) or a cross validation without tuning')
            print('      cvgridsearch | cv')
            print('-l <minsampleleafpercent> ')
            print('-d <maxdepth> ')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-u", "--sigmas_ufile"):
            sigmas_ufile = arg
        elif opt in ("-f", "--fold"):
            fold = int(arg)
        elif opt in ("-m", "--method"):
            method = arg
        elif opt in ("-t", "--methodtune"):
            method_tune = arg
        elif opt in ("-a", "--action"):
            action = arg
        elif opt in ("-p", "--passe"):
            passe = int(arg)
        elif opt in ("-s", "--step"):
            step = int(arg)
        elif opt in ("-l", "--minsampleleafpercent"):
            min_samples_leaf_percent = float(arg)
        elif opt in ("-d", "--maxdepth"):
            if int(arg) == 0:
                max_depth = None
            else:
                max_depth = int(arg)

    #Lecture data
    #loading data set ozone after perturbations
    X = pd.read_csv(inputfile, sep = '\t',index_col=0)
    #Convert pandas into nd array
    X=X.values

    Y = pd.read_csv(outputfile, sep = '\t',index_col=0)
    #Convert pandas into nd array
    Y=Y.values
    Y=Y.ravel()

    #Decoupage
    X_train_fold5, X_fold5 = train_test_split(X, test_size = 0.2, random_state=42)# 80 et 20 pourcents
    Y_train_fold5, Y_fold5 = train_test_split(Y, test_size = 0.2, random_state=42)

    X_train_123, X_fold4 = train_test_split(X_train_fold5, test_size = 0.25, random_state=42)#25 prcts de 80 prcts
    Y_train_123, Y_fold4 = train_test_split(Y_train_fold5, test_size = 0.25, random_state=42)
    X_train_fold4 = np.concatenate((X_train_123, X_fold5), axis=0)
    Y_train_fold4 = np.concatenate((Y_train_123, Y_fold5), axis=0)

    X_train_12, X_fold3 = train_test_split(X_train_123, test_size = 0.33, random_state=42)#33 prcts de 60 prcts
    Y_train_12, Y_fold3 = train_test_split(Y_train_123, test_size = 0.33, random_state=42)
    X_train_45 = np.concatenate((X_fold4, X_fold5), axis=0)
    Y_train_45 = np.concatenate((Y_fold4, Y_fold5), axis=0)
    X_train_fold3 = np.concatenate((X_train_12, X_train_45), axis=0)
    Y_train_fold3 = np.concatenate((Y_train_12, Y_train_45), axis=0)

    X_fold1, X_fold2 = train_test_split(X_train_12, test_size = 0.5, random_state=42)#50 prcts de 40 prcts
    Y_fold1, Y_fold2 = train_test_split(Y_train_12, test_size = 0.5, random_state=42)
    X_train_345 = np.concatenate((X_fold3, X_train_45), axis=0)
    Y_train_345 = np.concatenate((Y_fold3, Y_train_45), axis=0)
    X_train_fold2 = np.concatenate((X_fold1, X_train_345), axis=0)
    Y_train_fold2 = np.concatenate((Y_fold1, Y_train_345), axis=0)

    X_train_fold1 = np.concatenate((X_fold2, X_train_345), axis=0)
    Y_train_fold1 = np.concatenate((Y_fold2, Y_train_345), axis=0)

    print("shape de X_fold1 ", np.shape(X_fold1), "shape de Y_fold1 ", np.shape(Y_fold1), "shape de X_train_fold1 ", np.shape(X_train_fold1),  "shape de Y_train_fold1 ", np.shape(Y_train_fold1))
    print("shape de X_fold2 ", np.shape(X_fold2), "shape de Y_fold2 ", np.shape(Y_fold2),  "shape de X_train_fold2 ", np.shape(X_train_fold2),  "shape de Y_train_fold2 ", np.shape(Y_train_fold2))
    print("shape de X_fold3 ", np.shape(X_fold3), "shape de Y_fold3 ", np.shape(Y_fold3),  "shape de X_train_fold3 ", np.shape(X_train_fold3),  "shape de Y_train_fold3 ", np.shape(Y_train_fold3))
    print("shape de X_fold4 ", np.shape(X_fold4), "shape de Y_fold4 ", np.shape(Y_fold4),  "shape de X_train_fold4 ", np.shape(X_train_fold4),  "shape de Y_train_fold4 ", np.shape(Y_train_fold4))
    print("shape de X_fold5 ", np.shape(X_fold5), "shape de Y_fold5 ", np.shape(Y_fold5),  "shape de X_train_fold5 ", np.shape(X_train_fold5),  "shape de Y_train_fold5 ", np.shape(Y_train_fold5))

    if fold == 1:
        X_test_fold = X_fold1
        X_train_fold = X_train_fold1
        Y_test_fold = Y_fold1
        Y_train_fold = Y_train_fold1
    elif fold == 2:
        X_test_fold = X_fold2
        X_train_fold = X_train_fold2
        Y_test_fold = Y_fold2
        Y_train_fold = Y_train_fold2
    elif fold == 3:
        X_test_fold = X_fold3
        X_train_fold = X_train_fold3
        Y_test_fold = Y_fold3
        Y_train_fold = Y_train_fold3
    elif fold == 4:
        X_test_fold = X_fold4
        X_train_fold = X_train_fold4
        Y_test_fold = Y_fold4
        Y_train_fold = Y_train_fold4
    elif fold == 5:
        X_test_fold = X_fold5
        X_train_fold = X_train_fold5
        Y_test_fold = Y_fold5
        Y_train_fold = Y_train_fold5

    #Action
    if action == "cvgridsearch":
        # uncertain_trees_process.py -i Xp_ozone_VC_ss_min.csv -o Y_oz.csv -f 1 -m Tmse -t TmsePredprob -a cvgridsearch -p 0 -s 5 -l 0.1 -d 0
        print("[LOG] -----------")
        print("[LOG] Gridsearch | multithreading ", mt)
        cvgridsearch(method, method_tune, X, X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, passe, step, min_samples_leaf_percent, max_depth)
    elif action == "cv":
        # uncertain_trees_process.py -i Xp_ozone_VC_ss_min.csv -o Y_oz.csv -u sigmas_u.csv -f 1 -m Tmse -a cv -l 0.1 -d 0
        print("[LOG] -----------")
        if sigmas_ufile == 'EmptyFile':
            print("[LOG] Cross-validation")
            fname = 'sigmas_u_cv.csv'
            if os.path.isfile(fname):
                sigmas_u = pd.read_csv(fname, sep = '\t',index_col=0)
                sigmas_u=sigmas_u.values
                sigmas_u=sigmas_u.ravel()
                crossvalidation(sigmas_u, method, X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, min_samples_leaf_percent, max_depth)
            else :
                print("[LOG] The file \"sigmas_u_cv.csv\" does not exist in the directory.")
        else :
            print("[LOG] Cross-validation with sigmas_u given as a script parameter ")
            if os.path.isfile(sigmas_ufile):
                sigmas_u = pd.read_csv(sigmas_ufile, sep = '\t',index_col=0)
                sigmas_u=sigmas_u.values
                sigmas_u=sigmas_u.ravel()
                crossvalidation(sigmas_u, method, X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, min_samples_leaf_percent, max_depth)
            else :
                print("[LOG] The sigmas_u file ", sigmas_ufile, " does not exist in the directory.")
            
        

print("[LOG] End of script")
print("[LOG] -----------")
