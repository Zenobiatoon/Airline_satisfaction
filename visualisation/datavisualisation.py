# -*- coding: utf-8 -*-
"""datavisualisation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JDbA0gyKu0JLUSx5pjJJwLa4_7oEIMb3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("satisfaction.csv", sep = ";")
df.head()

df.info()

df.describe()

#print(df['Seat comfort'].value_counts())

#affichage de la frequence des variables ordinales 
frequencies={}
for i in df.columns:
  frequencies[i]= df[i].value_counts()
plotdata= pd.DataFrame(frequencies)
plotdata.head()

plotdata.iloc[0:5,8:13]

plotdata.iloc[0:6,8:-3].plot(kind='barh', stacked=True,figsize=(15, 8))
plt.title('fréquence des variables ordinales appariées')
plt.show()

plotdata.iloc[0:6,8:-2].transpose().plot(kind='barh', stacked=True,figsize=(15, 8))
plt.title('fréquence des variables ordinales appariées')
plt.show()

#Relation entre age du passager et la classe en fonction du type de voyage
sns.catplot(x='Type of Travel',y='Age',hue='Class',kind='bar',data=df)
plt.title('Relation entre age du passager et la classe en fonction du type de voyage')

#Relation entre age du passager et satifaction en fonction du type de voyage
sns.catplot(x='satisfaction_v2',y='Age',hue='Type of Travel',kind='bar',data=df)
plt.title('Relation entre age du passager et satifaction en fonction du type de voyage')

#Relation entre distance  du vol  et satifaction en fonction du type de voyage
sns.catplot(x='satisfaction_v2',y='Flight Distance',hue='Type of Travel',kind='bar',data=df)
plt.title('Relation entre distance  du vol  et satifaction en fonction du type de voyage')

#Etude de la relation entre distance du vol et satifaction en fonction de la classe
sns.catplot(x='satisfaction_v2',y='Flight Distance',hue='Class',kind='bar',data=df)
plt.title('Relation entre distance du vol et satifaction en fonction de la classe')

#affichage d'un PairGrid 
g= sns.PairGrid(df[['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes','satisfaction_v2']], hue='satisfaction_v2')
g.map(plt.scatter, alpha=0.6)
g.add_legend()

#afficher un graphique de correlation entre les variables numeriques 
df.satisfaction_v2 = df.satisfaction_v2.apply(lambda x: 1 if x == 'satisfied' else 0 )
df.Gender = df.Gender.apply(lambda x: 1 if x == 'Male' else 0 )
corr = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=True)
plt.title (' graph de correlation entre toutes les donnees')
plt.show()

#proportion de clients satisfaits/ pas satisfaits
df['satisfaction_v2'].value_counts()
sns.countplot(df['satisfaction_v2'])
plt.title ( 'Proportion de clients satisfaits/ pas satisfaits')