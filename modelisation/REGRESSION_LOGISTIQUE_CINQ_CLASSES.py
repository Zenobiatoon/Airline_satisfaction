"""
================================================================
     PREPROCESSING + MODELISATION AVEC LA REGRESSION LOGISTIQUE
     + DECOUPAGE DES NOTES DE SATISFACTION EN 5 CLASSES
================================================================
"""

# Import des librairies
import time
import pandas as pd
import numpy as np
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

# Preprocessing des modalités 0 des variables de notation : affectation à la moyenne
liste_var=list(df.columns[7:21])
for i in liste_var:
    df[i]=np.where(df[i]==0,np.around(df[i].mean()),df[i])

# Preprocessing des variables discrètes : transformation avec get_dummies
df = pd.get_dummies(df,columns = ['Gender'],prefix = 'Gen')
df = pd.get_dummies(df,columns = ['Customer Type'],prefix = 'Cust_typ')
df = pd.get_dummies(df,columns = ['Type of Travel'],prefix = 'Type_o_t')
df = pd.get_dummies(df,columns = ['Class'],prefix = 'Class')
for i in liste_var:
    df = df.join(pd.get_dummies(df[i], prefix=i))

# Preprocessing de la variable Age : découpage en quantiles
age_quantiles = pd.qcut(df['Age'], 4, labels=False)
age_quantiles=age_quantiles.rename('Age_cut')

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
clf_lr_v0= LogisticRegressionCV(solver='liblinear', random_state=123) # par défaut LBFGS => Pas de convergence
clf_lr_v0.fit(X_train, y_train)

# Prédiction + matrice de confusion + classfication report + score
y_pred = clf_lr_v0.predict(X_test)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm, '\n')
print(classification_report(y_test, y_pred))
print('Le score sur Train est de :',clf_lr_v0.score(X_train, y_train))
print('Le score sur Test est de :',clf_lr_v0.score(X_test, y_test))

# Calcul des probabilités d'appartenir à la classe 0 ou 1
probs = clf_lr_v0.predict_proba(X_test)

# Construction de la courbe ROC et calcul de l'AUC
fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)
print('L\'indice AUC est de :',roc_auc)

plt.plot(fpr, tpr, color='orange', lw=2, label='Modèle clf (auc = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux faux positifs')
plt.ylabel('Taux vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show();


