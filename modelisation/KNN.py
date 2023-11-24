"""
============================================================
     PREPROCESSING + MODELISATION AVEC L'ALGORITHME KNN
============================================================
"""

# Import des librairies
import time
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Import du fichier de données
df=pd.read_excel(r'C:/Users/PC/Desktop/DATASCIENTEST/2-PROJET/1-DATA/INPUT/SOURCE/satisfaction.xlsx')
df.head(2)

# Preprocessing : suppression des variables non pertinentes
df.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Flight Distance'], axis=1, inplace=True)

# Preprocessing des variables de notation : transformation en 2 classes indicatrices
df.replace([1,2,3], 0, inplace=True)
df.replace([4,5], 1, inplace=True)

# Preprocessing des variables discrètes : transformation avec get_dummies
df = pd.get_dummies(df,columns = ['Gender'],prefix = 'Gen')
df = pd.get_dummies(df,columns = ['Customer Type'],prefix = 'Cust_typ')
df = pd.get_dummies(df,columns = ['Type of Travel'],prefix = 'Type_o_t')
df = pd.get_dummies(df,columns = ['Class'],prefix = 'Class')
# Preprocessing de la variable Age : découpage en quantiles
age_quantiles = pd.qcut(df['Age'], 4, labels=False)
age_quantiles=age_quantiles.rename('Age_cut')

# Preprocessing de la variable discrète Age : transformation avec get_dummies
df=df.join(pd.DataFrame(age_quantiles))
df= pd.get_dummies(df,columns = ['Age_cut'],prefix = 'Age')
df['target']=df['satisfaction_v2'].apply(lambda x : 1 if x=='satisfied' else 0)

# Séparation des données et de la cible
target=df['target']
data=df.iloc[:,3:-1]

# %%
# KNN SIMPLE
# ===========

# Décomposition des données en deux ensembles : entraînement et test
X_train, X_test, y_train, y_test=train_test_split(data,target,test_size=0.2, random_state=123 )

# Chargement du classifieur et construction du modèle sur les données d'entraînement
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)

# Calcul des prédictions + matrice de confusion
y_pred = knn.predict(X_test)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm, '\n')
# score avec la distance de Minkowski  + classfication report
score_minkowski_test = knn.score(X_test, y_test)
print('le score minkowski sur Test est de :',score_minkowski_test)
print(classification_report(y_test, y_pred))

# %%
# GRID SEARCH KNN
# ===============

st = time.process_time()

clf_knn = neighbors.KNeighborsClassifier()
parametres = {'n_neighbors': range(2,10)}
clf_knn_grid = GridSearchCV(estimator=clf_knn, param_grid=parametres,cv=3)
clf_knn_grid.fit(X_train, y_train)

et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')

# Les meilleurs paramètres
print('Meilleurs paramètres :', clf_knn_grid.best_params_)
print('Le score sur le Test', clf_knn_grid.score(X_test, y_test))

# Calcul des prédictions + matrice de confusion + classification report
y_pred = clf_knn_grid.predict(X_test)
pd.crosstab(y_test, y_pred, rownames = ['Classe réelle'], colnames = ['Classe prédite par knn'])
print(classification_report(y_test, y_pred))