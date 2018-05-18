
import sklearn
from sklearn import tree
print(sklearn.__path__)
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf)

reg1 = tree.DecisionTreeRegressor(criterion="mae")
reg1 = reg1.fit(X,Y)
print(reg1)
#but = voir si le print s afficher

reg = tree.DecisionTreeRegressor(criterion="mse2")
reg = reg.fit(X,Y)
print(reg)
#but = voir si la classe clone qu on appelle fonctionne

X1 = [[0, 0], [2, 2]]
y1 = [0.5, 2.5]
reg3 = tree.DecisionTreeRegressor(criterion="mse2")
reg3 = reg3.fit(X1, y1)
print(reg3.predict([[1, 1]]))
#Renvoit bien la meme chose que sklearn du net pr criterion="mae"

