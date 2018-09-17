
# tree notes

La construction de l'abre suis la logique suivante:

Les modéles type CART sont chargés depuis des apelles python sklearn:

    cart_model = load_cart_model_from_sklearn([models parameters...])

On fit ensuite les modéles en passant les données:

    cart_model.fit(X, y)

La méthode `fit` suit le processus suivant quelque soit le modéle d'arbre choisis:

* La méthode `build` de la classe [DepthFirst|BreadthFirst] définis dand le fichier `_tree.pyx` est le point d'entrée,
* cette methode contruit un arbre `Tree` dont la structure est aussi définis dans `_tree.pyx`,
* la méthode de split passée en paramètre parmis [BestSplitter|Bestsplitter2|RandomSplitter] seras utilisé pour le split. Noté que l'on n'est pas obligé de passer cette option. Pour les criètre classique, l'arbre est construit en depth-first et pour `mseprobe` en breadth-first,
* les splitter sont définis dans `_splitter.pyx`,
* les splitters utilisent un critérion [mse|mseprob|etc] qui sont définit dans `_criterion.pyx` (également passé en paramètres) (mge ou gini par default)


Les splitters ont 3 méthodes clés appelées pendant la construsction de l'arbre définis dans la méthode principale `build`:

* init() initialise le spliter
* node_reset() initialise la position du splitter et les buffer du critérion
* node_split() split une node en fonction du critèrion donné.

La méthode node_split est la méthode clé qui va faire appelle au méthode du critérion tel que `citérion_update` `critérion_impurity` etc...


Une fois la méthode fit complété, l'object `_tree` de type `Tree` définis dans `_tree.pyx` est gardé en mémoire et contient les informations de l'arbre permettant de faire la prédiciton de nouvelles observations. La méthode predict appellé en python fais en fait apelle à la méthode predict de la classe Tree (`_tree.predict()`).

La construction et comparaison de mse de la baseline et la construction dynamique sont exposé dans le fichier `test2.py`.

## Quantile

* un changement à l'initialisation (nouveau paramètre alpha, appel à la librairie statsmodels et QuantReg pour une régression res_qr = QuantReg(y, P).fit(alpha) et enfin l'initialisation de gamma_chapeau par  res_qr.params)
* ligne 29 où gamma_chapeau est  calculé par le même procédé mais avec P_test (res_qr = QuantReg(y, P_test).fit(alpha) puis res_qr.params)
* ligne 30 la fonction de loss L() devient la fonction de loss quantile L_alpha. Ce qui se traduit par un procédé de ce type :
si y_i-P[i,].gamma_chapeau >0  alors F_test[i] = alpha abs(y_i - P[i,].gamma_chapeau)
sinon, F_test[i] =  (1-alpha) abs(y_i - P[i,].gamma_chapeau)
et au final, F_test=1/n \sum_i=1^n F_test[i]
* ligne 51 gamma_chapeau est actualisé selon le même procédé faisant appel à QuantReg de statsmodels avec la matrice P retenue.
