"""
================================================================
     PREPROCESSING + MODELISATION AVEC LA REGRESSION LOGISTIQUE
     + DECOUPAGE DES NOTES DE SATISFACTION EN 2 CLASSES
================================================================
"""

# Import des librairies
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
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
# RL SIMPLE
# ===========
# Décomposition des données en deux ensembles : entraînement et test
X_train, X_test, y_train, y_test=train_test_split(data,target,test_size=0.2, random_state=123 )

# Création du classifieur et construction du modèle sur les données d'entraînement
clf_lr_v0= LogisticRegressionCV(random_state=123) # par défaut LBFGS

# Prédiction + matrice de confusion + classfication report + score
clf_lr_v0.fit(X_train, y_train)
y_pred = clf_lr_v0.predict(X_test)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm, '\n')
print(classification_report(y_test, y_pred))
print('Le score sur Train est de :',clf_lr_v0.score(X_train, y_train))
print('Le score sur Test est de :',clf_lr_v0.score(X_test, y_test))

# Calcul des probabilités d'appartenir à la classe 0 ou 1
probs = clf_lr_v0.predict_proba(X_test)
y_preds = np.where(probs[:,1]>0.4,1,0)

# Construction de la courbe ROC et calcul de l'AUC
fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)
print('L\'indice AUC est de :',roc_auc)

# Courbe ROC
plt.plot(fpr, tpr, color='orange', lw=2, label='Modèle clf (auc = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux faux positifs')
plt.ylabel('Taux vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show();

# %%
# RL GRID SEARCH
# ==============
st = time.process_time()

clf_lr_grid = LogisticRegression(random_state=123,max_iter=1000)
params_lr = {'solver': ['liblinear', 'lbfgs'], 'C': [10**(i) for i in range(-4, 3)]}
gridcv = GridSearchCV(clf_lr_grid, param_grid=params_lr, scoring='accuracy', cv=3)
gridcv.fit(X_train, y_train)

et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')

pd.DataFrame(gridcv.cv_results_)[['params', 'mean_test_score', 'std_test_score']]

print('La meilleure combinaison d\'hyperparamètres :',gridcv.best_params_)

# On relance le modèle avec les hyperparamètres sélectionnés sur 3 échantillons stratifiés (cv) :
best_clf_grid=LogisticRegression(solver='liblinear', C=0.1)
best_clf_grid.fit(X_train, y_train)

# Prédiction + matrice de confusion
y_pred = best_clf_grid.predict(X_test)
cm=pd.crosstab(y_test, y_pred, rownames = ['Classe réelle'], colnames = ['Classe prédite par rf'])
print(cm, '\n')

# Calcul des scores
train_acc = accuracy_score(y_true=y_train, y_pred=best_clf_grid.predict(X_train))
test_acc = accuracy_score(y_true=y_test, y_pred=best_clf_grid.predict(X_test))

print('Les paramètres du modèle :',gridcv.best_params_)
print('Le score sur Train :',train_acc)
print('Le score sur Test :',test_acc)
print(classification_report(y_test, y_pred))


