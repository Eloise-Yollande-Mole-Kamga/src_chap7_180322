import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from math import *

 

#==================================================================================================
# RESEAU SEQUENTIAL
# définir un reseau "sequantial" (= complet) pour resoudre un probleme de regression : 
# ici notre but est de faire apprendre au réseau les entrées du problème de production et retourne une valeur réelle 
#qui est le cout de production
#
# on decoupe cet exemple en deux fonctions principales : une 1ere qui construit le reseau et realise
# l'apprentissage et une seconde qui charge le reseau prealablement sauvegarde et le teste sur de nouvelles donnees
#=================================================================================================



# Fonction de creation de la premiere couche du réseau de neurones
def MLP_LN_regression_construct_first_couche(model, activation, kernel_initializer, input_shape, Nb_neurones):
	# Creation d'une couche de Nb_neurones neurones avec un input_shape=(input_shape,) car les vecteurs en entrée ont input_shape composantes
	model.add(keras.layers.Dense(Nb_neurones, activation=activation, kernel_initializer=kernel_initializer, input_shape=(input_shape,)))

# Fonction de creation d'un couche cachee du réseau de neurones
def MLP_LN_regression_construct_hidden_couche(model, activation, Nb_neurones):
	model.add(keras.layers.Dense(Nb_neurones, activation=activation))

# Fonction de creation de la derniere couche du réseau de neurones
def MLP_LN_regression_construct_last_couche(model, Nb_neurones):
	# Creation d'une couche de sortie de Nb_neurones neurones
	model.add(keras.layers.Dense(Nb_neurones))

# construction du reseau "sequential"
def MLP_LN_regression_construct(Nb_composantes_input):
	print(Nb_composantes_input)
	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on construit un reseau sequantiel

	model = keras.Sequential()
	
	# premiere "vraie" couche de 8 neurones avec un input_shape=(4,) car les vecteurs en entrée ont 4 composantes
	MLP_LN_regression_construct_first_couche(model,'relu', 'normal', Nb_composantes_input, 8)#80

	# on ajoute deux couches l'une de  4 neurones et l'autre de 2 neurones
	nb_neuron = [4, 2]
	#nb_neuron = [80, 80]
	for i in range(0, 1):
		MLP_LN_regression_construct_hidden_couche(model, 'relu', nb_neuron[i])

	# couche de sortie : 1 seul neurone car on a un problème de regression : on souhaite generer une valeur a partir des entrees
	# on utilise la fonction d'activation par défaut (linear) car la sortie peut être positive ou négative et est non bornée
	MLP_LN_regression_construct_last_couche(model, 1)

	# affichage du model à l'écran et dessin du modele dans le fichier model.png
	model.summary()
	keras.utils.plot_model(model, 'modelSequential.png', show_shapes=True)

	return model

def MLP_LN_regression_FctError_Apprentissag_SaveModel(model, optimizer, loss, X_data, Y_data, epochs, batch_size, verbose):

	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  on definit la fonction d'erreur et on fait l'apprentissage


	# compile the model
	# The mean squared error (mse) loss is minimized when fitting the model.
	model.compile(optimizer=optimizer, loss=loss)
	#loss = fonction d'erreur


	# apprentissage sur le jeu de données :
	X_data_ = np.asarray(X_data)
	Y_data_ = np.asarray(Y_data)
	history = model.fit(X_data_, Y_data_, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=verbose)

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("erreur en fonction des époques")
	plt.ylabel('loss (carré moyen des erreurs)')
	plt.xlabel('nombre d\'époques')
	plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
	plt.savefig('loss_sequentialModel.pdf')
	plt.show()

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  sauvegarde du modele

	#on sauvegarde le modele dans un fichier texte pour pouvoir l'utiliser plus tard sur des données
	model.save('model_LN_reg.h5')

# la fonction suivante teste le modele créé et entrainé sur les données de test
def MLP_LN_regression_predire(Nb_vect_test, X_test, y_test, nom_fichier):

	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on charge le modele sauvegarde precedemment 

	model = keras.models.load_model('model_LN_reg.h5')

	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  on genere aleatoirement de nouvelles donnees et on calcule l'erreur faite par le modele

	
	y = y_test # vecteur qui va stocker la "vraie" solution
	y_predict = [] # vecteur qui va stocker la prediction du reseau de neurones
	err = 0

	N = Nb_vect_test # nb de vecteurs a tester
	#print(X_test)
	y_predict=model.predict(X_test) # on calcule la prediction faite par le reseau de neurones
	for i in range(N):
		#print(i)
		#print((X_test[i]))
		#y_predict.append(a)
		#print(y[i])
		#print(y_predict[i])
		err = err + (abs(y[i] - y_predict[i]) * 100) / y[i] # on calcule l'erreur realisee par le reseau de neurones


	# affichage de l'erreur
	print(err / N)
	print("erreur_"+nom_fichier+ "= " + str(err / N) + "%")


	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  visualisation graphique du resultat

	x = np.linspace(1,N,N) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	
	y, y_predict =zip(*sorted(zip(y, y_predict)))

	Nb_valtest=N
	if(nom_fichier=='SequentialModel_prediction_train.pdf'):
		y_by_100=[]#y_by_100 est la liste obtenues après transformation de la liste y_train_temp en la regroupant par paquet de 100 
		Nb_valtest_by_100=Nb_valtest//100
		x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
		for k in range(Nb_valtest_by_100-1):
			y_by_100.append(np.mean(y[100*k:100*k+100]))
		y_by_100.append(np.mean(y[5900:5909]))
		
		y_predict_by_100=[]#y_by_100 est la liste obtenues après transformation de la liste y en la regroupant par paquet de 100 #x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
		for k in range(Nb_valtest_by_100-1):
			y_predict_by_100.append(np.mean(y_predict[100*k:100*k+100]))
		y_predict_by_100.append(np.mean(y_predict[5900:5909]))
	else:
		y_predict_by_100=y_predict
		y_by_100=y


	plt.plot(x, y_predict_by_100, "r", label="Coût de production prédit par le réseau de neurone")
	plt.plot(x, y_by_100, "g", label="Coût de production optimal" )
	plt.ylabel('coût de production')
	plt.xlabel('numéro de l\'instance')
	plt.legend()
	plt.savefig(nom_fichier)
	plt.show()




	##Ordonne par ordre croissant les valeurs optimales
	#y_and_y_predict = [[0] * N for i in range(2)]
	#y_predict_temp = y_predict
	#y_temp = y
	#y_pred_list= y_predict_temp.tolist()
	#y_pred_list_temp = y_pred_list
	#for i in range(0, N):
	#	y_and_y_predict[0][i]=min(y_temp) #la plus petite valeur de y
	#	mole= y_temp.index(min(y_temp))
	#	y_and_y_predict[1][i]=y_pred_list_temp[y_temp.index(min(y_temp))][0] #le y_predict qui correspond à la plus petite valeur de y
	#	y_pred_list_temp.remove(y_pred_list_temp[y_temp.index(min(y_temp))])
	#	y_temp.pop(y_temp.index(min(y_temp))) #suppression du plus petit élément de la liste
	##print ("y_and_y_predict : \n", y_and_y_predict)


	##Dessine des courbes des valeurs optimales et des valeurs predites
	#print("La longueur du vecteur de solution est ", len(y_and_y_predict[0]))
	#i=0;
	#while(i<=len(y_and_y_predict[0])/15):
	#	plt.plot(x[i:i+15],y_and_y_predict[0][i:i+15], "g", label="Opt", marker='o')
	#	plt.plot(x[i:i+15],y_and_y_predict[1][i:i+15], "r", label="Predict", marker='v')
	#	plt.legend()
	#	plt.xlabel("vecteurs")
	#	plt.ylabel("valeurs")
	#	i_ch = "%d" % i
	#	plt.savefig('prediction_courbe_' + i_ch + '.png')
	#	plt.show()
	#	i+=15
	
	##Dessine un nuage de points des valeurs optimales et des valeurs predites
	#i=0;
	#while(i<=len(y_and_y_predict[0])/15):
	#	plt.scatter(x[i:i+15],y_and_y_predict[0][i:i+15],color="green", s = 10,label="Opt", marker='o') # on affiche en verts les valeurs des "vraies" solutions pour chaque vecteur
	#	plt.scatter(x[i:i+15],y_and_y_predict[1][i:i+15],color="red", s = 10, label="Predict", marker='v') # on affiche en rouge les valeurs predites par le modele
	#	plt.legend()
	#	plt.xlabel("vecteurs")
	#	plt.ylabel("valeurs")
	#	i_ch = "%d" % i
	#	plt.savefig('prediction_nuage_' + i_ch + '.png')
	#	plt.show()
	#	i+=15

	

	