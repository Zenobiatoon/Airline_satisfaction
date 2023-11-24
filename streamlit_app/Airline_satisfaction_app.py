import pandas as pd
import numpy as np
import streamlit as st
import matplotlib as plt
from PIL import Image

st.sidebar.title('Sommaire')
pages = ['Introduction', 'Data Visualisation', 'Pre Processing', 'Modelisation', 'Conclusion']
page = st.sidebar.radio('allez vers', pages)

if page == pages[0]:
    st.title('Airline Satisfaction')
    st.subheader(':blue[_By Zineb, Ndiaye and Sophia_ ]')
    df = pd.read_csv('satisfaction.csv', sep=';')
    st.header('Introduction')

    image = Image.open("airport.jpg")
    st.image(image)

    st.write(
        'La présente étude s’intéresse à la « satisfaction du client » suite à un voyage effectué sur un des vols d’une compagnie aérienne américaine. ')
    st.write(
        "Il s’agit d’un questionnaire envoyé aux personnes ayant voyagé à bord du vol en question. La base de données est disponible sur Kaggle mais les commentaires qui l'accompagnent ne donnent pas plus de détails sur la compagnie aérienne et/ou sur le dispositif de l’enquête.")
    st.write("Les objectifs de ce projet sont les suivants : ")
    st.write('● Analyser les données de satisfaction des clients')
    st.write('● Construire des modèles prédictifs de la satisfaction clients')
    st.write(
        '● Déterminer le/les éléments du parcours client qui apportent le plus de satisfaction (check-in, onboarding, etc.)')
    image =Image.open("parcours_client.png")
    st.image(image)

    st.write('Voici notre data frame satisfaction:')
    st.dataframe(df.head())


    st.write('Voici la volumétrie de notre jeu de données:')
    st.write('➔ :red[129.880] observations uniques correspondant aux répondants du vol')
    st.write('➔ :red[1] variable cible : _satisfaction_v2_ (‘satisfied’/‘neutral or dissatisfied’)')
    st.write('➔ :red[22] variables explicatives (ce qui est peu)')

    st.write('Nous avons relevé :')
    st.write(
        '➔ :red[4] variables catégorielles : Gender (M/F ), Customer type (Loyal/disloyal ), Type of travel(Business/Personal), Class(Business/Eco/Eco plus).')
    st.write(
        '➔ :red[14] variables quantitatives ordinales : ce sont les notes de satisfaction (0 à 5) données à chaque question (Seat comfort, Inflight entertainment, food and drink...). La valeur 0 étant l’équivalent du ‘pas concerné’')
    st.write(
        '➔ :red[4] variables quantitatives continues : Age, Flight distance (en miles), Departure delay (en minutes) , Arrival delay (en minutes).')

elif page == pages[1]:
    st.title('Data Visualisation')

    st.write(
        "Afin d'étudier la particularité de notre jeu de données, nous avons d’abord utilisé des fonctionnalités de la bibliothèque pandas, comme info(), describe(), ou bien head(). Ensuite à l'aide des librairies matplotlib et seaborn nous avons créé des graphiques afin de visualiser notre data et mieux comprendre nos variables explicatives. Enfin, à l'aide des fonctions corr() et V Cramer nous avons respectivement, établi les corrélations entre les variables numériques et les dépendances entre variables catégorielles.")

    st.subheader('Notes de satisfaction')
    image = Image.open('notesatis.jpg')
    st.image(image, width=400, use_column_width='always')

    st.write(
        'Les variables ci-dessus sont ordinales et reposent sur une échelle de notation allant de 1 à 5 (0 : ‘ne se prononce pas’). Nous sommes donc face à deux sujets : l’existence de non-réponses, et la nature de ces variables qui est numérique et ordinale à la fois. Il faudra choisir des règles de gestion appropriées lors du pré-processing.')

    st.subheader('Les variables continues')
    image = Image.open('grpah3.jpg')
    st.image(image, width=400, use_column_width='always')
    image = Image.open('graph5.jpg')
    st.image(image, width=400, use_column_width='always')

    st.write('Les analyses de la distribution n’ont pas montré d’anomalies sur les variables continues')

    st.subheader('Les variables catégorielles')
    image = Image.open('grpah5categ.jpg')
    st.image(image, width=400, use_column_width='always')
    image = Image.open('grapg6.jpg')
    st.image(image, width=400, use_column_width='always')

    st.subheader('Relation entre satifaction et retard de vol ')
    image = Image.open('graph1.jpg')
    st.image(image, width=300, use_column_width='always')

    st.write(
        'La satisfaction ou la non satisfaction du voyageur ne semble pas dépendre du retard du vol. Il faudra confirmer cette observation par une analyse statistique.')

    st.subheader('Matrice de corrélation de spearman des variables ordinales ')
    image = Image.open('corr.jpg')
    st.image(image, width=300, use_column_width='always')

    st.write(
        'La seule variable corrélée avec la satisfaction client est Inflight entertainment avec un taux de 0.52, ce qui est moyen.')

    st.subheader('Heatmap de V de cramer des variables categorielles')
    image = Image.open('v.jpg')
    st.image(image, width=300, use_column_width='always')

    st.subheader('Conclusion:')
    st.write(
        'La seule variable corrélée avec la satisfaction client est Inflight entertainment avec un taux de 0.52, ce qui est moyen.')
    st.write(
        'Notre projet de prédiction de la satisfaction client consiste à mettre en place un modèle de :red[classification]')
    st.write(
        'La variable cible (satisfaction) est fortement corrélée/liée à la variable Inflight entertainment, et à la variable Class.')
    st.write(
        'Les services qui influent le plus sur la satisfaction globale du client sont le service Online, la gestion des bagages, le divertissement à bord, la propreté dans l’appareil, l’espace dédié aux jambes et le service à bord.')

elif page == pages[2]:
    st.title('Pre Processing')
    st.write('Les données sont issues d\'un questionnaire et non d\'une base de données classique.')
    st.markdown('**:red[:warning: Il y a très peu de nettoyage à faire]**')

    st.write('**1. On vérifie le nombre de 0 :**')
    if st.checkbox('affichez les 0'):
        image = Image.open('nb_zero.png')
        st.image(image)

    st.write('**2. On binarise les notes de satisfaction :**')
    st.markdown(':bulb: *Les notes 0, 1, 2, 3 sont affectées à la valeur 0 et les notes 4 et 5 à la valeur 1*')


    st.write('**3. On vérifie les valeurs manquantes :**')
    if st.checkbox('affichez les VM'):
        df = pd.read_csv('satisfaction.csv', sep=';')
        st.dataframe(df.isna().sum())
    st.markdown(':bulb:*Seule variable concernée : Arrival Delay avec 0.3% de VM*')

    st.write('**4. On supprime les variables non pertinentes :** ')
    st.write('V Cramer(Departure delay x Satisfaction) : 0.10')
    st.write('V Cramer(Arrival delay x Satisfaction) : 0.08')
    st.write('V Cramer(Flight distance x Satisfaction) : 0.18')
    if st.checkbox('affichez les seuils d\'intensité'):
        image = Image.open('V Cramer.png')
        st.image(image, width=420)
    st.markdown(':bulb:*Les trois variables continues montrent un faible lien avec la satisfaction du passager mais aussi avec les autres variables*')
    st.write('**5. Que faire de la variable Age :** ')
    if st.checkbox('affichez le croisement'):
        image = Image.open('3-analyse_bivariee_age_satisfaction.png')
        st.image(image)
    st.markdown(':pencil: Les 39-60 ans sont **plus satisfaits** du vol que les jeunes et les + de 60 ans')
    st.markdown(':bulb:*Cette variable n\'est pas importante pour la classification (V Cramer : 0.17) mais peut l\'être pour le ciblage marketing*')

elif page == pages[3]:
   st.title('Modelisation')
   st.write('Nettoyage du data frame')
   st.write("Voici notre data frame satisfaction nettoyé: nous avons procédé à la suppression des variables non pertinentes, à la transformation de la variable cible en 2/5 classes indicatrices indiquant le niveau de satisfactions(satisfied/non-satisified),et à la discrétisation des variables carégorielles")
   df = pd.read_csv('satisfaction.csv', sep=';')   
   
   
       
   #Preprocessing : suppression des variables non pertinentes

   df=df.drop(columns=['Departure Delay in Minutes' ,'Arrival Delay in Minutes','Flight Distance'], axis=1)
    
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
   target_5= df['satisfaction_v2']#variable cible pour les modèles à cinq classes
   df['target']=df['satisfaction_v2'].apply(lambda x : 1 if x=='satisfied' else 0)
 
    
   st.dataframe(df.head())

    # Séparation des données et de la cible 
   target=df['target']
   data=df.iloc[:,3:-1]


    #Splitting the dataset into the Training set and Test set
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 0)

   st.write('Entrainement des modèles')
   model_choisi= st.selectbox(label="Choix du modèle",options=['Classifieur Naif bayésien','Support Vector Machine','Random Forest','K-Neighbors','Régression logistique - deux classes','Régression logistique - cinq classes'])
   
   from sklearn.naive_bayes import GaussianNB
   #from sklearn.ensemble import VotingClassifier, StackingClassifier
   #from sklearn.model_selection import train_test_split, KFold, cross_validate
   from sklearn.metrics import f1_score
   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn import svm, model_selection, preprocessing
   from sklearn.model_selection import GridSearchCV
   
   
   
   
   
   def train_model(model_choisi):
       if model_choisi=='Classifieur Naif bayésien':
           model_grid=GaussianNB()
       elif model_choisi=='Support Vector Machine':
           model_grid =svm.SVC(kernel='poly',gamma=0.1)
       elif model_choisi=='Random Forest':
           model= RandomForestClassifier(random_state=123)
           model_grid = GridSearchCV(estimator=model, param_grid={'max_features': ["sqrt", "log2", None ],
                                                         'min_samples_split': range(2, 40, 2)}, cv=3)
       elif model_choisi=='K-Neighbors':
     
           model = KNeighborsClassifier()
           parametres = {'n_neighbors': range(2,10)}
           model_grid = GridSearchCV(estimator=model, param_grid=parametres,cv=3)
       elif model_choisi=='Régression logistique - deux classes':
     
           model = LogisticRegression(random_state=123,max_iter=1000)
           params_lr = {'solver': ['liblinear', 'lbfgs'], 'C': [10**(i) for i in range(-4, 3)]}
           model_grid = GridSearchCV(model, param_grid=params_lr, scoring='accuracy', cv=3)
           
       elif model_choisi=='Régression logistique - cinq classes':
     
           model = LogisticRegression(random_state=123,max_iter=1000)
           params_lr = {'solver': ['liblinear', 'lbfgs'], 'C': [10**(i) for i in range(-4, 3)]}
           model_grid = GridSearchCV(model, param_grid=params_lr, scoring='accuracy', cv=3)
           X, X_t, y, y_t = train_test_split(data, target_5, test_size = 0.2, random_state = 0)
           
       if model_choisi=='Régression logistique - cinq classes':
          model_grid.fit(X,y)
          score= model_grid.score(X_t,y_t)
       else:
           model_grid.fit(X_train,y_train)
           score= model_grid.score(X_test,y_test)
       return (score)
   st.write("Score test",train_model(model_choisi))
   
   
   st.write('Optimisation des modèles par Voting et Stacking')
   method_choisi= st.selectbox(label="Choix de la méthode",options=['Voting','Stacking'])
   

   
   st.write('Ci-dessous les métriques obtenus sur les 4 modèles testés :')
   mage = Image.open('CaptureMODEL.png')
   st.image(mage, width=300, use_column_width='always')
   
   st.write('Analyse')
   st.write('Le modèle le plus performant reste :red[le Random Forest] qui est adapté pour résoudre simplement et rapidement un problème de classification. Le report de classification ')
   
      
   image = Image.open('RF_best_model.png')
   st.image(image, width= 300, use_column_width= 'always')

   def train_method(method_choisi):
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        
       #Algorithme de  Stacking
        if method_choisi=='Stacking':
            clf1= KNeighborsClassifier(n_neighbors=8)
            clf2=RandomForestClassifier(random_state=123,max_features = 'sqrt', min_samples_split = 20)
            clf3=svm.SVC(kernel='poly',gamma=0.1)
            method= StackingClassifier( estimators=[('knn', clf1), ('rf', clf2), ('SVM', clf3)])
        #Algorithme de  Voting
        if method_choisi=='Voting':
            clf1= KNeighborsClassifier(n_neighbors=8)
            clf2=RandomForestClassifier(random_state=123,max_features = 'sqrt', min_samples_split = 20)
            clf3=svm.SVC(kernel='poly',gamma=0.1)
            method= VotingClassifier( estimators=[('knn',clf1),('rf',clf2),('svm',clf3)], voting= 'hard')
        
        method.fit(X_train,y_train)
        score=method.score(X_test,y_test)
        return (score)
   st.write("Score test",train_method(method_choisi))
       
elif page == pages[4]:
    st.title('Conclusion')

    st.write('➔ Le score obtenu etant de :red[0.90], on peut conclure que le modèle détécté :red[correctement] le niveau de satisfaction des clients.')
    st.write("➔ Il est important de souligner que la majorité des personnes ayant répondu à l'étude effectuent un :red[voyage professionnel] ce qui va influencer le niveau de satisfaction globale (même si nous ne sommes pas en présence d'un jeu de données déséquilibré).")
    st.write("➔ Ce système est un outil important pour les aéroports afin d'améliorer leur service client et savoir quelle étape est influente. Ce projet peut servir de base pour développer un système de recommandation pour les aéroports. ")


    st.subheader('Regard critique')
    st.write("Si nous avions du temps en plus, nous aurions proposer une gestion différente des valeurs manquantes à savoir :red[le filtrage collaboratif]")
    st.write(" De plus, nous aurions effectué davantage de croisements de variable")

    image = Image.open('imageair.jpg')
    st.image(image, width= 300, use_column_width= 'always')




