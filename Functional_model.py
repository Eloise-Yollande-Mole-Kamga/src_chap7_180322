import tensorflow.keras as keras
import random as rd
import numpy as np
import matplotlib.pyplot as plt


from math import *

#==================================================================================================
#                         
#-----------------------------------------------------------------------------------------------
#  RESEAU FUNCTIONAL
#
# on definit ici un reseau "FUNCTIONAL" (= on definit chaque couche et on definit lesquelles sont reliees entre elles)
# pour resoudre un probleme de regression : 
# 
#ici notre but est de faire apprendre au réseau les entrées du problème de production et retourne une valeur réelle 
#qui est le cout de production
#========================================================================================

def reseauFunctional(X_train1, X_train2, y_train, X_train1_shape, X_train2_shape, X_test1, X_test2, y):
	#Les donnees d'entrainements : X_train1, X_train2, y_train
	#Les donnees de test : X_test1, X_test2, y
	#Le nombre de colones de chaque paquet d'inputs: X_train1_shape, X_train2_shape

	# on veut apprendre au reseau a calculer le cout de production
	# puis on les concatene avec une couche "concat" pour avoir une sortie unique
		
	#---------------------------------------------------------------------------------------------------
	#  ETAPE 1. creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	my_input1 = keras.Input(shape=(X_train1_shape,), name="input1") 
	my_input2 = keras.Input(shape=(X_train2_shape,), name="input2")  

	#defintion des couches dans le premier sous-reseau 
	#chaque couche recoit en parametre la couche precedente a laquelle elle est reliee
	couche1_m1 = keras.layers.Dense(8, activation='selu')(my_input1)
	couche2_m1 = keras.layers.Dense(4, activation='selu')(couche1_m1) 
	couche3_m1 = keras.layers.Dense(1, activation='linear')(couche2_m1)

	#defintion des couches dans le second sous-reseau 
	couche1_m2 = keras.layers.Dense(10, activation='selu')(my_input2)
	couche2_m2 = keras.layers.Dense(5, activation='selu')(couche1_m2)
	couche3_m2 = keras.layers.Dense(1, activation='linear')(couche2_m2)

	# on concatene les couches finales des 2 sous-reseaux
	merge = keras.layers.concatenate([couche3_m1, couche3_m2])

	#on finit avec une couche avec un seul neurone
	couchefinal = keras.layers.Dense(1, activation='linear')(merge)
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2], outputs=couchefinal)

	#dessin du modele
	model.summary()
	keras.utils.plot_model(model, "reseauFunctional.pdf", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  ajout de la fonction d erreur et apprentissage


	model.compile(optimizer='adam', loss='mean_squared_error')

	
	#on met les donnees d'entrainement dans le bon format
	X_train1 = np.asarray(X_train1)
	X_train2 = np.asarray(X_train2)
	y_train = np.asarray(y_train)
	print(X_train1.shape)
	print(X_train2.shape)
	#entrainement
	history = model.fit({"input1": X_train1, "input2": X_train2}, y_train, validation_split=0.33, epochs=200, batch_size=16, verbose=0)
	
	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("erreur en fonction des epochs")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('loss_functionalModel.pdf')
	plt.show()



	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  Test du reseau avec de nouvelles donnees


	# make a prediction
	y_predict = [] #prediction
	err = 0

	N = len(X_test1) #nb de vecteurs a tester

	y_predict = model.predict({"input1": X_test1, "input2": X_test2}) # valeur predite
	y_predict_train=model.predict({"input1": X_train1, "input2": X_train2}) 
	for i in range(N):
		err = err + (abs(y[i] - y_predict[i]) * 100) / y[i] 

	#affichage de l'erreur
	print(err / N)
	print("erreur = " + str(err / N) + "%")

	#representation graphique du resultat
	#x = np.linspace(1,N,N)
	#plt.scatter(x, y, color="green", s = 10, label='solution')
	#plt.scatter(x, y_predict, color="red", s = 10, label='prediction')
	#plt.xlabel('exemples')
	#plt.ylabel('valeurs')
	#plt.legend()
	
	#plt.show( )

	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  visualisation graphique du resultat

	
	N=len(X_test1)
	x = np.linspace(1,N,N) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	y, y_predict =zip(*sorted(zip(y, y_predict)))
	plt.plot(x, y, "g", label="Opt : Pipe" )
	plt.plot(x, y_predict, "r", label="Predict")
	plt.legend()
	plt.savefig('functionalModel_prediction_test.pdf')
	plt.show()

	N=len(X_train1)
	x = np.linspace(1,N,N) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	y_train, y_predict_train =zip(*sorted(zip(y_train, y_predict_train)))
	plt.plot(x, y_train, "g", label="Opt : Pipe" )
	plt.plot(x, y_predict_train, "r", label="Predict")
	plt.legend()
	plt.savefig('functionalModel_prediction_train.pdf')
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

	#x = np.linspace(1,N,N) #on a N valeurs sur l'axe des x (une pour chaque vecteur)

	##Dessine des courbes des valeurs optimales et des valeurs predites
	#i=0;
	#while(i<=len(y_and_y_predict[0])/15):
	#	plt.plot(x[i:i+15],y_and_y_predict[0][i:i+15], "g", label="Opt", marker='o')
	#	plt.plot(x[i:i+15],y_and_y_predict[1][i:i+15], "r", label="Predict", marker='v')
	#	plt.legend()
	#	plt.xlabel("exemples")
	#	plt.ylabel("valeurs")
	#	i_ch = "%d" % i
	#	plt.savefig('prediction_courbe_' + i_ch + '.pdf')
	#	plt.show()
	#	i+=15
	
	##Dessine un nuage de points des valeurs optimales et des valeurs predites
	#i=0;
	#while(i<=len(y_and_y_predict[0])/15):
	#	plt.scatter(x[i:i+15],y_and_y_predict[0][i:i+15],color="green", s = 10,label="Opt", marker='o') # on affiche en verts les valeurs des "vraies" solutions pour chaque vecteur
	#	plt.scatter(x[i:i+15],y_and_y_predict[1][i:i+15],color="red", s = 10, label="Predict", marker='v') # on affiche en rouge les valeurs predites par le modele
	#	plt.legend()
	#	plt.xlabel("exemples")
	#	plt.ylabel("valeurs")
	#	i_ch = "%d" % i
	#	plt.savefig('prediction_nuage_' + i_ch + '.pdf')
	#	plt.show()
	#	i+=15
