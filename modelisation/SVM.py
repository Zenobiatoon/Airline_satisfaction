"""
================================================================
     PREPROCESSING + MODELISATION AVEC L'ALGORITHME SVM
=================================================================
"""

# Import des librairies
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm

df=pd.read_excel(r'C:/Users/PC/Desktop/DATASCIENTEST/2-PROJET/1-DATA/INPUT/SOURCE/satisfaction.xlsx')
df.head()

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

# Décomposition des données en deux ensembles : entraînement et test
X_train, X_test, y_train, y_test=train_test_split(data,target,test_size=0.2, random_state=123 )

# Création du classifieur et construction du modèle sur les données d'entraînement
st = time.process_time()

clf_svm = svm.SVC(gamma=0.01,  kernel='poly')
clf_svm.fit(X_train, y_train)

et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')

# Prédiction + matrice de confusion
y_pred = clf_svm.predict(X_test)
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

# Calcul des scores + classfication report
st = time.process_time()
print('Le score sur Train est de :', clf_svm.score(X_train, y_train))
print('Le score sur Test est de :', clf_svm.score(X_test, y_test))
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')

print(classification_report(y_test, y_pred))

