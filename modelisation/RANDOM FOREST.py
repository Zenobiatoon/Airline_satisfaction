"""
============================================================
     PREPROCESSING + MODELISATION AVEC L'ALGORITHME
                     RANDOM FOREST
============================================================
"""

# Import des librairies
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# RF SIMPLE
# ===========
# Décomposition des données en deux ensembles : entraînement et test
X_train, X_test, y_train, y_test=train_test_split(data,target,test_size=0.2, random_state=123 )

# Création du classifieur et construction du modèle sur les données d'entraînement
clf_rf_v0 = RandomForestClassifier(random_state=123)
clf_rf_v0.fit(X_train, y_train)

# Prédiction + matrice de confusion + classfication report + score
y_pred = clf_rf_v0.predict(X_test)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm, '\n')
print(classification_report(y_test, y_pred))
print('Le score sur Train est de :',clf_rf_v0.score(X_train, y_train))
print('Le score sur Test est de :',clf_rf_v0.score(X_test, y_test))

# Bootstrap 1000 : Création du classifieur et construction du modèle sur les données d'entraînement
clf_rf_v0 = RandomForestClassifier(random_state=123, bootstrap=1000)
clf_rf_v0.fit(X_train, y_train)

# Bootstrap 1000 : Prédiction + matrice de confusion + classfication report + score
y_pred = clf_rf_v0.predict(X_test)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

# Prédiction + matrice de confusion + classfication report + score
print(cm, '\n')
print(classification_report(y_test, y_pred))
print('Le score sur Train est de :',clf_rf_v0.score(X_train, y_train))
print('Le score sur Test est de :',clf_rf_v0.score(X_test, y_test))

# %%
# RF GRID SEARCH
# ==============
# Chargement du classifieur et construction du modèle sur les données d'entraînement
clf_rf = RandomForestClassifier(random_state=123)
clf_rf_grid = GridSearchCV(estimator=clf_rf, param_grid={'max_features': ["sqrt", "log2", None ],
                                                         'min_samples_split': range(2, 40, 2)}, cv=3)

st = time.process_time()
clf_rf_grid.fit(X_train, y_train)
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')

pd.DataFrame(clf_rf_grid.cv_results_)[['params', 'mean_test_score', 'std_test_score']]

print('La meilleure combinaison d\'hyperparamètres :',clf_rf_grid.best_params_)