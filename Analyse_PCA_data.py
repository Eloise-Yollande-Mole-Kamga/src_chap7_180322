import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Source : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_ACP_Python.pdf

def preparation_data(X):
    #Preparation des donnees
    #Nous devons explicitement centrer et réduire les variables pour réaliser une ACP normée avec PCA. 
    #Nous utilisons la classe StandardScaler pour ce faire.

    #instanciation 
    sc = StandardScaler()

    #transformation – centrage-réduction
    Z = sc.fit_transform(X)
    print(Z)

    #Vérifions, par acquit de conscience, les propriétés du nouvel ensemble de données.
    # Les moyennes sont maintenant nulles (aux erreurs de troncature près) et les ecarts-type unitaires.
    
    #moyenne
    print(np.mean(Z,axis=0))
    
    #écart-type
    print(np.std(Z,axis=0,ddof=0))

    return Z

def analyse_PCA(X,Z,n):
    #Il faut instancier l'objet PCA dans un premier temps, nous affichons ses propriétés.
    acp = PCA(n_components=2, svd_solver='full')
    #affichage des paramètres
    print(acp)
    #Le paramètre (svd_solver = 'full') indique l'algorithme utilisé pour la décomposition
    # en valeurs singulières. Nous choisissons la méthode 'exacte', sélectionnée de toute 
    # manière par défaut pour l'appréhension des bases de taille réduite. D'autres approches
    # sont disponibles pour le traitement des grands ensembles de données. Le nombre de
    # composantes (K) n'étant pas spécifié (n_components = None), il est par défaut égal au nombre de variables.
    
    #Nous pouvons lancer les traitements dans un second temps. La fonction fit_transform() renvoie en sortie les 
    #coordonnées factorielles F_ik que nous collectons dans la variable coord
    #calculs
    coord = acp.fit_transform(Z)
    #nombre de composantes calculées 
    print(acp.n_components_)

    #PCA fournit également les proportions de variance associées aux axes.
    #proportion de variance expliquée 
    print(acp.explained_variance_ratio_)

    #Si le resultat est [0.73680968 0.14267705 0.06217768 0.03565368 0.01546687 0.00721505] alors l'interprétation est
    #La première composante accapare 73.68% de l’information disponible. Il y a un fort ‘’effet taille’’ dans nos données.
    # Nous disposons de 87.94% avec les deux premiers facteurs. Les suivants semblent anecdotiques.

    #les variances (valeurs propres, λk) associées aux axes factoriels :
    print(acp.singular_values_**2/n)

    #nuage de points
    print("co*************************************************************")
    print(type(coord))
    print(np.shape(coord))
    XX=[]
    Y=[]
    for i in range(n):
        XX.append(coord[i,0])
        Y.append(coord[i,1])
    plt.scatter(XX,Y)


    #nuage de points avec label
    #positionnement des individus dans le premier plan 
    #fig, axes = plt.subplots(figsize=(12,12))#
    #axes.set_xlim(-6,6) #même limites en abscisse#
    #axes.set_ylim(-6,6)# #et en ordonnée #placement des étiquettes des observations
    #for i in range(n): #
        #nom_inst = "inst_" + "%d" % i
        #plt.annotate(nom_inst,(coord[i,0],coord[i,1]))
        #plt.annotate('.',(coord[i,0],coord[i,1]))#
    #ajouter les axes 
    #plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)#
    #plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)#

    
    #affichage
    plt.savefig('Analyse_PCA_' + "%d" % n + '.pdf')
    plt.show()