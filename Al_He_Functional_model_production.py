import tensorflow.keras as keras
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import *


#définition de la fonction de calcul de l'erreur

def creationPoidsSetup(N):
	biais = np.zeros(N)
	poids = np.eye(N)#matrice identite

	#-1 sur les cases (i,i+1)
	for i in range(0,N-1):
		poids[i][i+1] = -1

	return(poids, biais)

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
#def my_loss_gap(y_true, y_pred):
#	return ((y_true- y_pred)/y_true)*100

def compteNbPoids(model):

	trainable_count = np.sum([Kbackend.count_params(w) for w in model.trainable_weights])
	non_trainable_count = np.sum([Kbackend.count_params(w) for w in model.non_trainable_weights])

	print('Total params: {:,}'.format(trainable_count + non_trainable_count))
	print('Trainable params: {:,}'.format(trainable_count))
	print('Non-trainable params: {:,}'.format(non_trainable_count))
	 

#affiche les poids et biais de la couche dont le nom est passe en parametre
def affichePoids(model, name):
	message_couche="affichage des poids de la couche "+ name
	print ("affichage des poids de la couche", name)
	print("\n")
	#afficher les poids qui arrive sur une couche
	WS = model.get_layer(name).get_weights()

	print("weights setup = ", WS[0] )
	print("biais setup = ", WS[1] )
	nom_fichier="weights_"+name+".txt"
	with open(nom_fichier, "w+") as file:
		file.write(message_couche)
		wei="weights setup\n"
		file.write(wei)
		for line in WS[0]:
			np.savetxt(file, line, fmt='%.2f')
		bia="biais setup\n"
		file.write(bia)
		#for line in WS[1]:
		np.savetxt(file, WS[1], fmt='%.2f')
#constructiion du réseau Al_He

def calculCout(last,N):
	#N = len(last)
	return np.dot(last, np.arange(N))


def CustomLoss(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train):
	#N = nombre de périodes max

	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on genere des donnees aleatoirement 

	#Cv_train = []
	#Cf_train = []
	#Rend_train = []
	#lambda_train = []
	#arange_train = []

	#prod_train = []
	#lastRefuel_train = []
	#setup_train = []
	#y_train = []


	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	#on a 4 vecteurs de 5 composantes en entree
	my_input1 = keras.Input(shape=(N,), name="input1") 
	my_input2 = keras.Input(shape=(N,), name="input2")  
	my_input3 = keras.Input(shape=(N,), name="input3") 
	my_input4 = keras.Input(shape=(N+2,), name="input4")  
	my_input5 = keras.Input(shape=(N, ), name="input5")  #entree fictive qui nous permet de passer des coeff pour faire le prod. scalaire


	#on concatene les input1 et input 2 puis input3 et input4
	concat1 = keras.layers.concatenate([my_input1, my_input2])
	concat2 = keras.layers.concatenate([my_input3, my_input4])

	# on cree deux couches sigmoides de taille N (une par couche concat)
	couche1_A = keras.layers.Dense(N, activation='sigmoid')(concat1)
	couche1_B = keras.layers.Dense(N, activation='sigmoid')(concat2)


	#on concatee les couches precedentes
	concat3 = keras.layers.concatenate([couche1_A, couche1_B])

	#on cree les couches de "présortie"
	pre_prod = keras.layers.Dense(N, activation='sigmoid', name="out1")(concat3) #  gamma
	pre_lastRefuel = keras.layers.Dense(N, activation='softmax', name="out2")(concat3)#  gamma*
	#pre_setup = keras.layers.Dense(N, activation='sigmoid', name="out3")(pre_prod)# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes

	#Ajouter ceci si pour passer de la couche prod a la couche setup il faut un biais nul et des poids (-1,1) fixes et enlever la ligne précédente
	(poids_setup, biais_setup) = creationPoidsSetup(N)
	pre_setup = keras.layers.Dense(N, activation='relu', name="out_setup", weights=[poids_setup, biais_setup], trainable=False)(pre_prod) # tau
	

	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	out_prod = tf.keras.layers.Multiply()([my_input1, pre_prod])
	out_setup = tf.keras.layers.Multiply()([my_input2, pre_setup])
	out_lastRefuel = tf.keras.layers.Multiply()([my_input5, pre_lastRefuel])

	#on concatene les couches et on fait la somme de toutes les composantes
	concat4 = keras.layers.concatenate([out_prod, out_setup, out_lastRefuel])
	out_val = keras.layers.Dense(1, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(concat4)



	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2, my_input3, my_input4, my_input5], outputs=out_val)

	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauCustomLoss.pdf", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  ajout de la fonction d erreur et apprentissage

	print("debut compile")
	model.compile(optimizer='adam', loss='mean_squared_error')
	#print("fin compile")
	#print("debut transforme data")
	#on met les donnees d'entrainement dans le bon format
	#print(Cv_train)

	Cv_train2 = np.asarray(Cv_train[90:]) #Les 90 premières instances seront les instances de test 
	#print("debut transforme Cf_train2")
	Cf_train2 = np.asarray(Cf_train[90:])#[90:]
	#print("debut transforme Rend_train2")
	Rend_train2 = np.asarray(Rend_train[90:])#[90:]
	#print("debut transforme lambda_train2")
	lambda_train2 = np.asarray(lambda_train[90:])#[90:]
	#print("milieu transforme data")
	#prod_train = np.asarray(prod_train)
	#lastRefuel_train = np.asarray(lastRefuel_train)
	#setup_train = np.asarray(setup_train)
	arange_train2 = np.asarray(arange_train[90:])#[90:]
	y_train2 = np.asarray(y_train[90:])#[90:]
	#print("fin transforme data")

	

	#entrainement
	#print (Cv_train2.shape)
	#print (type(Cv_train2))
	#print (Cf_train2.shape)
	#print (type(Cf_train2))
	#print (Rend_train2.shape)
	#print (type(Rend_train2))
	#print (lambda_train2.shape)
	#print (type(lambda_train2))
	#print (arange_train2.shape)
	#print (type(arange_train2))
	#print (y_train2.shape)
	#print (type(y_train2))
	#print("debut fit")
	history = model.fit({"input1": Cv_train2, "input2": Cf_train2, "input3":Rend_train2, "input4":lambda_train2, "input5": arange_train2},
		  y_train2,validation_split=0.33,
		 epochs=200, batch_size=16, verbose=0)
	#print("fin fit")

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("erreur en fonction des époques")
	plt.ylabel('loss (carré moyen des erreurs)')
	plt.xlabel('nombre d\'époques')
	plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
	plt.savefig('loss_prediction_courbe_Al_He_.pdf')
	plt.show()
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 5.  on teste

	print (Cv_train2.shape)
	print (type(Cv_train2))

	print (Cv_train2[0:1].shape)
	print (type(Cv_train2[0:1]))

	y = model.predict({"input1" : np.asarray(Cv_train[0:90]) , "input2":  np.asarray(Cf_train[0:90]),
				   "input3": np.asarray(Rend_train[0:90]), "input4": np.asarray(lambda_train[0:90]),
				   "input5": np.asarray(arange_train[0:90])})
	print ("predit : ", y)
	print("theorique", y_train[0:90])
	Nb_valtest=len(y_train[0:90])
	err=0
	for i in range(Nb_valtest):
		#print(i)
		#print((X_test[i]))
		#y_predict.append(a)
		#print(y_train[i])
		#print(y[i])
		err = err + (abs(y_train[i] - y[i]) * 100) / y_train[i] # on calcule l'erreur realisee par le reseau de neurones


	# affichage de l'erreur
	print(err / Nb_valtest)
	print("erreur = " + str(err / Nb_valtest) + "%")


	#représentation graphique en une unique courbe
	y_train_temp=y_train[0:90]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g", label="Opt" )
	plt.plot(x, y, "r", label="Predict")
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet_test.pdf')
	plt.show()

	y = model.predict({"input1" : np.asarray(Cv_train[90:]) , "input2":  np.asarray(Cf_train[90:]),
				   "input3": np.asarray(Rend_train[90:]), "input4": np.asarray(lambda_train[90:]),
				   "input5": np.asarray(arange_train[90:])})
	print ("predit : ", y)
	print("theorique", y_train[90:])
	Nb_valtest=len(y_train[90:])

	#représentation graphique en une unique courbe
	y_train_temp=y_train[90:]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g", label="Opt" )
	plt.plot(x, y, "r", label="Predict")
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet_train.pdf')
	plt.show()


	#représentation graphique en une uniqueplusieurs courbes
	#x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	#i=0;
	#while(i<=Nb_valtest):
	#	plt.plot(x[i:i+15],y_train[i:i+15], "g", label="Opt", marker='o')
	#	plt.plot(x[i:i+15],y[i:i+15], "r", label="Predict", marker='v')
	#	plt.legend()
	#	plt.xlabel("vecteurs")
	#	plt.ylabel("valeurs")
	#	i_ch = "%d" % i
	#	plt.savefig('prediction_courbe_Al_He_' + i_ch + '.png')
	#	plt.show()
	#	i+=15
	

def CustomLoss_with_drop_out(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train,dropOutRate):
	#N = nombre de périodes max

	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on genere des donnees aleatoirement 

	#Cv_train = []
	#Cf_train = []
	#Rend_train = []
	#lambda_train = []
	#arange_train = []

	#prod_train = []
	#lastRefuel_train = []
	#setup_train = []
	#y_train = []


	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	#on a 4 vecteurs de 5 composantes en entree
	my_input1 = keras.Input(shape=(N,), name="input1") 
	my_input2 = keras.Input(shape=(N,), name="input2")  
	my_input3 = keras.Input(shape=(N,), name="input3") 
	my_input4 = keras.Input(shape=(N+2,), name="input4")  
	my_input5 = keras.Input(shape=(N, ), name="input5")  #entree fictive qui nous permet de passer des coeff pour faire le prod. scalaire


	#on concatene les input1 et input 2 puis input3 et input4
	concat1 = keras.layers.concatenate([my_input1, my_input2])
	concat2 = keras.layers.concatenate([my_input3, my_input4])

	# on cree deux couches sigmoides de taille N (une par couche concat)
	couche1_A = keras.layers.Dense(N, activation='sigmoid')(concat1)
	couche1_A = keras.layers.Dropout(rate=dropOutRate, name="droupout1A")(couche1_A)
	couche1_B = keras.layers.Dense(N, activation='sigmoid')(concat2)
	couche1_B = keras.layers.Dropout(rate=dropOutRate, name="droupout1B")(couche1_B)


	#on concatee les couches precedentes
	concat3 = keras.layers.concatenate([couche1_A, couche1_B])
	concat3_drop = keras.layers.Dropout(rate=dropOutRate, name="droupout1A_1B")(concat3)

	#on cree les couches de "présortie"
	pre_prod = keras.layers.Dense(N, activation='sigmoid', name="out1")(concat3_drop) #  gamma
	pre_prod = keras.layers.Dropout(rate=dropOutRate, name="droupout_pre_prod")(pre_prod)
	pre_lastRefuel = keras.layers.Dense(N, activation='softmax', name="out2")(concat3_drop)#  gamma*
	pre_lastRefuel = keras.layers.Dropout(rate=dropOutRate, name="droupout_prelast_fuel")(pre_lastRefuel)
	#pre_setup = keras.layers.Dense(N, activation='sigmoid', name="out3")(pre_prod)# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes

	#Ajouter ceci si pour passer de la couche prod a la couche setup il faut un biais nul et des poids (-1,1) fixes et enlever la ligne précédente
	(poids_setup, biais_setup) = creationPoidsSetup(N)
	pre_setup = keras.layers.Dense(N, activation='relu', name="out_setup", weights=[poids_setup, biais_setup], trainable=False)(pre_prod) # tau
	pre_setup = keras.layers.Dropout(rate=dropOutRate, name="presetu_up")(pre_setup)

	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	out_prod = tf.keras.layers.Multiply()([my_input1, pre_prod])
	out_setup = tf.keras.layers.Multiply()([my_input2, pre_setup])
	out_lastRefuel = tf.keras.layers.Multiply()([my_input5, pre_lastRefuel])

	#on concatene les couches et on fait la somme de toutes les composantes
	concat4 = keras.layers.concatenate([out_prod, out_setup, out_lastRefuel])
	concat4 = keras.layers.Dropout(rate=dropOutRate, name="concat_out")(concat4)
	out_val = keras.layers.Dense(1, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(concat4)
	



	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2, my_input3, my_input4, my_input5], outputs=out_val)

	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauCustomLoss.png", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  ajout de la fonction d erreur et apprentissage

	print("debut compile")
	model.compile(optimizer='adam', loss='mean_squared_error')
	print("fin compile")
	print("debut transforme data")
	#on met les donnees d'entrainement dans le bon format
	#print(Cv_train)

	Cv_train2 = np.asarray(Cv_train[90:]) #Les 90 premières instances seront les instances de test
	print("debut transforme Cf_train2")
	Cf_train2 = np.asarray(Cf_train[90:])
	print("debut transforme Rend_train2")
	Rend_train2 = np.asarray(Rend_train[90:])
	print("debut transforme lambda_train2")
	lambda_train2 = np.asarray(lambda_train[90:])
	print("milieu transforme data")
	#prod_train = np.asarray(prod_train)
	#lastRefuel_train = np.asarray(lastRefuel_train)
	#setup_train = np.asarray(setup_train)
	arange_train2 = np.asarray(arange_train[90:])
	y_train2 = np.asarray(y_train[90:])
	print("fin transforme data")

	

	#entrainement
	print (Cv_train2.shape)
	print (type(Cv_train2))
	print (Cf_train2.shape)
	print (type(Cf_train2))
	print (Rend_train2.shape)
	print (type(Rend_train2))
	print (lambda_train2.shape)
	print (type(lambda_train2))
	print (arange_train2.shape)
	print (type(arange_train2))
	print (y_train2.shape)
	print (type(y_train2))
	print("debut fit")
	history = model.fit({"input1": Cv_train2, "input2": Cf_train2, "input3":Rend_train2, "input4":lambda_train2, "input5": arange_train2},
		  y_train2,
		 epochs=200, batch_size=16, verbose=0)
	print("fin fit")
	

	#affichage du poids des arcs du reseau de neurones
	#model.get_weights()
	affichePoids(model, "dense_1")#couche1_A
	affichePoids(model, "dense")#couche1_B
	#affichePoids(model, "droupout1A_AB" )#concat3_drop
	affichePoids(model, "out1")#pre_prod
	#affichePoids(model, "out2")#pre_lastRefuel
	affichePoids(model, "out_setup")#pre_setup
	affichePoids(model, "out_val")#out_val


	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.title("erreur en fonction des epochs")
	plt.savefig('loss_prediction_courbe_Al_He_.png')
	plt.show()
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 5.  on teste

	print (Cv_train2.shape)
	print (type(Cv_train2))

	print (Cv_train2[0:1].shape)
	print (type(Cv_train2[0:1]))

	y = model.predict({"input1" : np.asarray(Cv_train[0:90]) , "input2":  np.asarray(Cf_train[0:90]),
				   "input3": np.asarray(Rend_train[0:90]), "input4": np.asarray(lambda_train[0:90]),
				   "input5": np.asarray(arange_train[0:90])})
	print ("predit : ", y)
	print("theorique", y_train[0:90])
	Nb_valtest=90

	#représentation graphique en une unique courbe
	y_train_temp=y_train[0:90]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g", label="Opt" )
	plt.plot(x, y, "r", label="Predict")
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet.png')
	plt.show()
	
	
	#x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	#i=0;
	#while(i<=Nb_valtest):
	#	plt.plot(x[i:i+15],y_train[i:i+15], "g", label="Opt", marker='o')
	#	plt.plot(x[i:i+15],y[i:i+15], "r", label="Predict", marker='v')
	#	plt.legend()
	#	plt.xlabel("vecteurs")
	#	plt.ylabel("valeurs")
	#	i_ch = "%d" % i
	#	plt.savefig('prediction_courbe_Al_He_' + i_ch + '.png')
	#	plt.show()
	#	i+=15



#Construction du réseau de neurones Al_He qui calcule uniquement le cout de production en utilisant les valeurs y_i et y_i*, ici on n'enlève la dernière couche qui calculent i*tau_i
def CustomLoss_uniquement_prod(N, Cv_train, Cf_train, Rend_train, lambda_train, y_train,kkk):#, arange_train
	#N = nombre de périodes max

	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on genere des donnees aleatoirement 

	#Cv_train = []
	#Cf_train = []
	#Rend_train = []
	#lambda_train = []
	#arange_train = []

	#prod_train = []
	#lastRefuel_train = []
	#setup_train = []
	#y_train = []


	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	#on a 4 vecteurs de 5 composantes en entree
	my_input1 = keras.Input(shape=(N,), name="input1") 
	my_input2 = keras.Input(shape=(N,), name="input2")  
	my_input3 = keras.Input(shape=(N,), name="input3") 
	my_input4 = keras.Input(shape=(N+2,), name="input4")  
	##my_input5 = keras.Input(shape=(N, ), name="input5")  #entree fictive qui nous permet de passer des coeff pour faire le prod. scalaire


	#on concatene les input1 et input 2 puis input3 et input4
	concat1 = keras.layers.concatenate([my_input1, my_input2])
	concat2 = keras.layers.concatenate([my_input3, my_input4])
	
	# on cree deux couches sigmoides de taille N (une par couche concat)
	couche1_A = keras.layers.Dense(N, activation='sigmoid')(concat1)
	#dropout
	#couche1_A=keras.layers.Dropout(rate=0.8, name="concat_out")(couche1_A)
	couche1_B = keras.layers.Dense(N, activation='sigmoid')(concat2)
	#dropout
	#couche1_B=keras.layers.Dropout(rate=0.8, name="concat_out2")(couche1_B)



	#on concatee les couches precedentes
	concat3 = keras.layers.concatenate([couche1_A, couche1_B])

	#on cree les couches de "présortie"
	pre_prod = keras.layers.Dense(N, activation='sigmoid', name="out1")(concat3) #  gamma
	#dropout
	#pre_prod=keras.layers.Dropout(rate=0.8, name="concat_out3")(pre_prod)

	##pre_lastRefuel = keras.layers.Dense(N, activation='softmax', name="out2")(concat3)#  gamma*
	#pre_setup = keras.layers.Dense(N, activation='sigmoid', name="out3")(pre_prod)# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes

	#Ajouter ceci si pour passer de la couche prod a la couche setup il faut un biais nul et des poids (-1,1) fixes et enlever la ligne précédente
	(poids_setup, biais_setup) = creationPoidsSetup(N)
	pre_setup = keras.layers.Dense(N, activation='relu', name="out_setup", weights=[poids_setup, biais_setup], trainable=False)(pre_prod) # tau
	

	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	out_prod = tf.keras.layers.Multiply()([my_input1, pre_prod])
	out_setup = tf.keras.layers.Multiply()([my_input2, pre_setup])
	##out_lastRefuel = tf.keras.layers.Multiply()([my_input5, pre_lastRefuel])

	#on concatene les couches et on fait la somme de toutes les composantes
	concat4 = keras.layers.concatenate([out_prod, out_setup])#, out_lastRefuel
	out_val = keras.layers.Dense(1, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(concat4)



	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2, my_input3, my_input4], outputs=out_val)#, my_input5

	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauCustomLoss.png", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  ajout de la fonction d erreur et apprentissage

	print("debut compile")
	model.compile(optimizer='adam', loss='mean_squared_error')#mean_absolute_error
	print("fin compile")
	print("debut transforme data")
	#on met les donnees d'entrainement dans le bon format
	#print(Cv_train)

	Cv_train2 = np.asarray(Cv_train[90:]) #Les 90 premières instances seront les instances de test 
	#print("debut transforme Cf_train2")
	Cf_train2 = np.asarray(Cf_train[90:])#[90:]
	#print("debut transforme Rend_train2")
	Rend_train2 = np.asarray(Rend_train[90:])#[90:]
	#print("debut transforme lambda_train2")
	lambda_train2 = np.asarray(lambda_train[90:])#[90:]
	#print("milieu transforme data")
	#prod_train = np.asarray(prod_train)
	#lastRefuel_train = np.asarray(lastRefuel_train)
	#setup_train = np.asarray(setup_train)
	##arange_train2 = np.asarray(arange_train[90:])#[90:]
	y_train2 = np.asarray(y_train[90:])#[90:]
	print("fin transforme data")

	

	#entrainement
	#print (Cv_train2.shape)
	#print (type(Cv_train2))
	#print (Cf_train2.shape)
	#print (type(Cf_train2))
	#print (Rend_train2.shape)
	#print (type(Rend_train2))
	#print (lambda_train2.shape)
	#print (type(lambda_train2))
	##print (arange_train2.shape)
	##print (type(arange_train2))
	#print (y_train2.shape)
	#print (type(y_train2))
	print("debut fit")
	history = model.fit({"input1": Cv_train2, "input2": Cf_train2, "input3":Rend_train2, "input4":lambda_train2},
		  y_train2, validation_split=0.33,
		 epochs=200, batch_size=32, verbose=0)#, "input5": arange_train2
	print("fin fit")

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("erreur en fonction des époques")
	plt.ylabel('loss (carré moyen des erreurs)')
	plt.xlabel('nombre d\'époques')
	plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
	plt.savefig('loss_prediction_courbe_Al_He_.pdf')
	plt.show()


	#---------------------------------------------------------------------------------------------------
	# ETAPE 5.  on teste

	#print (Cv_train2.shape)
	#print (type(Cv_train2))

	#print (Cv_train2[0:1].shape)
	#print (type(Cv_train2[0:1]))

	y = model.predict({"input1" : np.asarray(Cv_train[0:90]) , "input2":  np.asarray(Cf_train[0:90]),
				   "input3": np.asarray(Rend_train[0:90]), "input4": np.asarray(lambda_train[0:90])})#,"input5": np.asarray(arange_train[0:90])
	#print ("predit : ", y)
	#print("theorique", y_train[0:90])
	
	Nb_valtest=len(y_train[0:90])
	err=0
	for i in range(Nb_valtest):
		#print(i)
		#print((X_test[i]))
		#y_predict.append(a)
		#print(y_train[i])
		#print(y[i])
		#y[i]=y[i]*kkk
		#y_train[i]=y_train[i]*kkk
		err = err + (abs(y_train[i] - y[i]) * 100) / y_train[i] # on calcule l'erreur realisee par le reseau de neurones


	# affichage de l'erreur
	print(err / Nb_valtest)
	print("erreur test = " + str(err / Nb_valtest) + "%")

	#représentation graphique en une unique courbe
	y_train_temp=y_train[0:90]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y, "r", label="Coût de production prédit par le réseau de neurones")
	plt.plot(x, y_train_temp, "g", label="Coût de production optimal" )
	plt.ylabel('coût de production')
	plt.xlabel('numéro de l\'instance')
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet_test.pdf')
	plt.show()

	y = model.predict({"input1" : np.asarray(Cv_train[90:]) , "input2":  np.asarray(Cf_train[90:]),
				   "input3": np.asarray(Rend_train[90:]), "input4": np.asarray(lambda_train[90:])})#,"input5": np.asarray(arange_train[0:90])
	#print ("predit : ", y)
	#print("theorique", y_train[90:])
	
	Nb_valtest=len(y_train[90:])
	err=0
	k_itt=90
	for i in range(Nb_valtest):
		#print(i)
		#print((X_test[i]))
		#y_predict.append(a)
		#print(y_train[i])
		#print(y[i])
		#y[i]=y[i]*kkk
		#y_train[i]=y_train[i]*kkk
		err = err + (abs(y_train[k_itt] - y[i]) * 100) / y_train[k_itt] # on calcule l'erreur realisee par le reseau de neurones
		k_itt=k_itt+1

	# affichage de l'erreur
	print(err / Nb_valtest)
	print("erreur train = " + str(err / Nb_valtest) + "%")


	#for i in range(Nb_valtest):
	#	y[i]=y[i]*kkk
	#	y_train[i]=y_train[i]*kkk
	#représentation graphique en une unique courbe
	y_train_temp=y_train[90:]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)

	y_train_temp_by_100=[]#y_train_temp_by_100 est la liste obtenues après transformation de la liste y_train_temp en la regroupant par paquet de 100 
	Nb_valtest_by_100=Nb_valtest//100
	x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur
	for k in range(Nb_valtest_by_100-1):
		y_train_temp_by_100.append(np.mean(y_train_temp[100*k:100*k+100]))
	y_train_temp_by_100.append(np.mean(y_train_temp[5900:5909]))

#y_by_100 est la liste obtenues après transformation de la liste y en la regroupant par paquet de 100 
	y_by_100=[]
	for k in range(Nb_valtest_by_100-1):
		y_by_100.append(np.mean(y[100*k:100*k+100]))
	y_by_100.append(np.mean(y[5900:5909]))
	
	plt.plot(x, y_by_100, "r", label="coût de production prédit par le réseau de neurones")
	plt.plot(x, y_train_temp_by_100, "g", label="Coût de production optimal" )
	plt.ylabel('coût de production')
	plt.xlabel('numéro de l\'instance')
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet_train.pdf')
	plt.show()




#Construction du réseau de neurones Al_He qui calcule uniquement le cout véhicule (date de dernière recharge) en utilisant la valeur i*tau_i, ici on n'enlève les deux dermières couches qui calculent y_i et y*_i	
def CustomLoss_uniquement_last_recharge(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train):
	#N = nombre de périodes max

	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on genere des donnees aleatoirement 

	#Cv_train = []
	#Cf_train = []
	#Rend_train = []
	#lambda_train = []
	#arange_train = []

	#prod_train = []
	#lastRefuel_train = []
	#setup_train = []
	#y_train = []


	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	#on a 4 vecteurs de 5 composantes en entree
	my_input1 = keras.Input(shape=(N,), name="input1") 
	my_input2 = keras.Input(shape=(N,), name="input2")  
	my_input3 = keras.Input(shape=(N,), name="input3") 
	my_input4 = keras.Input(shape=(N+2,), name="input4")  
	my_input5 = keras.Input(shape=(N, ), name="input5")  #entree fictive qui nous permet de passer des coeff pour faire le prod. scalaire


	#on concatene les input1 et input 2 puis input3 et input4
	concat1 = keras.layers.concatenate([my_input1, my_input2])
	concat2 = keras.layers.concatenate([my_input3, my_input4])

	# on cree deux couches sigmoides de taille N (une par couche concat)
	couche1_A = keras.layers.Dense(N, activation='sigmoid')(concat1)
	couche1_B = keras.layers.Dense(N, activation='sigmoid')(concat2)


	#on concatee les couches precedentes
	concat3 = keras.layers.concatenate([couche1_A, couche1_B])

	#on cree les couches de "présortie"
	##pre_prod = keras.layers.Dense(N, activation='sigmoid', name="out1")(concat3) #  gamma
	pre_lastRefuel = keras.layers.Dense(N, activation='softmax', name="out2")(concat3)#  gamma*
	#pre_setup = keras.layers.Dense(N, activation='sigmoid', name="out3")(pre_prod)# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes

	#Ajouter ceci si pour passer de la couche prod a la couche setup il faut un biais nul et des poids (-1,1) fixes et enlever la ligne précédente
	##(poids_setup, biais_setup) = creationPoidsSetup(N)
	##pre_setup = keras.layers.Dense(N, activation='relu', name="out_setup", weights=[poids_setup, biais_setup], trainable=False)(pre_prod) # tau
	

	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	##out_prod = tf.keras.layers.Multiply()([my_input1, pre_prod])
	##out_setup = tf.keras.layers.Multiply()([my_input2, pre_setup])
	out_lastRefuel = tf.keras.layers.Multiply()([my_input5, pre_lastRefuel])

	#on concatene les couches et on fait la somme de toutes les composantes
	##concat4 = keras.layers.concatenate([out_prod, out_setup, out_lastRefuel])
	out_val = keras.layers.Dense(1, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(out_lastRefuel)



	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2, my_input3, my_input4, my_input5], outputs=out_val)

	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauCustomLoss.png", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  ajout de la fonction d erreur et apprentissage

	print("debut compile")
	model.compile(optimizer='adam', loss='mean_squared_error')#mean_absolute_error
	print("fin compile")
	print("debut transforme data")
	#on met les donnees d'entrainement dans le bon format
	#print(Cv_train)

	Cv_train2 = np.asarray(Cv_train[90:]) #Les 90 premières instances seront les instances de test 
	#print("debut transforme Cf_train2")
	Cf_train2 = np.asarray(Cf_train[90:])#[90:]
	#print("debut transforme Rend_train2")
	Rend_train2 = np.asarray(Rend_train[90:])#[90:]
	#print("debut transforme lambda_train2")
	lambda_train2 = np.asarray(lambda_train[90:])#[90:]
	#print("milieu transforme data")
	#prod_train = np.asarray(prod_train)
	#lastRefuel_train = np.asarray(lastRefuel_train)
	#setup_train = np.asarray(setup_train)
	arange_train2 = np.asarray(arange_train[90:])#[90:]
	y_train2 = np.asarray(y_train[90:])#[90:]
	print("fin transforme data")

	

	##entrainement
	#print (Cv_train2.shape)
	#print (type(Cv_train2))
	#print (Cf_train2.shape)
	#print (type(Cf_train2))
	#print (Rend_train2.shape)
	#print (type(Rend_train2))
	#print (lambda_train2.shape)
	#print (type(lambda_train2))
	#print (arange_train2.shape)
	#print (type(arange_train2))
	#print (y_train2.shape)
	#print (type(y_train2))
	print("debut fit")
	history = model.fit({"input1": Cv_train2, "input2": Cf_train2, "input3":Rend_train2, "input4":lambda_train2, "input5": arange_train2},
		  y_train2, validation_split=0.33,epochs=200, batch_size=32, verbose=0)
	print("fin fit")

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.plot(history.history["val_loss"])
	plt.title("erreur en fonction des époques")
	plt.ylabel('loss (carré moyen des erreurs)')
	plt.xlabel('nombre d\'époques')
	plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
	plt.savefig('time_N20_loss_prediction_reseauALternativeLearning.pdf')
	#plt.savefig('loss_prediction_courbe_Al_He_.png')
	plt.show()
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 5.  on teste

	print (Cv_train2.shape)
	print (type(Cv_train2))

	print (Cv_train2[0:1].shape)
	print (type(Cv_train2[0:1]))

	y = model.predict({"input1" : np.asarray(Cv_train[0:90]) , "input2":  np.asarray(Cf_train[0:90]),
				   "input3": np.asarray(Rend_train[0:90]), "input4": np.asarray(lambda_train[0:90]),
				   "input5": np.asarray(arange_train[0:90])})
	Nb_valtest=len(y_train[0:90])
	err=0
	for i in range(Nb_valtest):
		#print(i)
		#print((X_test[i]))
		#y_predict.append(a)
		#print(y_train[i])
		#print(y[i])
		#y[i]=y[i]*kkk
		#y_train[i]=y_train[i]*kkk
		if(y_train[i]!=0):
			err = err + (abs(y_train[i] - y[i]) * 100) / y_train[i] # on calcule l'erreur realisee par le reseau de neurones


	# affichage de l'erreur
	print(err / Nb_valtest)
	print("erreur test = " + str(err / Nb_valtest) + "%")

	#print ("predit : ", y)
	#print("theorique", y_train[0:90])
	Nb_valtest=90#sur toutes les données d'entrainenment 2549
	#print("Le vecteur des valeurs optimales de la date de dernière recharge est \n")
	#print(y_train[0:90])
	#représentation graphique en une unique courbe
	y_train_temp=y_train[0:90]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g+", label="Numéro de période de la dernière recharge de la tournée optimale" )
	plt.plot(x, y, "ro", label="Numéro de période de la dernière recharge de la tournée prédite")
	plt.ylabel('Numéro de période de la dernière recharge')
	plt.xlabel('numéro de l\'instance')
	plt.legend()
	plt.savefig('time_N20_prediction_courbe_Al_He_test.pdf')
	plt.show()

	y = model.predict({"input1" : np.asarray(Cv_train[90:]) , "input2":  np.asarray(Cf_train[90:]),
				   "input3": np.asarray(Rend_train[90:]), "input4": np.asarray(lambda_train[90:]),
				   "input5": np.asarray(arange_train[90:])})
	Nb_valtest=len(y_train[90:])
	err=0
	k_itt=90
	for i in range(Nb_valtest):
		if(y_train[i]!=0):
			err = err + (abs(y_train[k_itt] - y[i]) * 100) / y_train[k_itt] # on calcule l'erreur realisee par le reseau de neurones
		k_itt=k_itt+1
	print("erreur train = " + str(err / Nb_valtest) + "%")


	y_train_temp=y_train[90:]
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)

	y_train_temp_by_100=[]#y_train_temp_by_100 est la liste obtenues après transformation de la liste y_train_temp en la regroupant par paquet de 100 
	Nb_valtest_by_100=Nb_valtest//100
	x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur
	for k in range(Nb_valtest_by_100-1):
		y_train_temp_by_100.append(np.mean(y_train_temp[100*k:100*k+100]))
	y_train_temp_by_100.append(np.mean(y_train_temp[5900:5909]))

#y_by_100 est la liste obtenues après transformation de la liste y en la regroupant par paquet de 100 
	y_by_100=[]
	for k in range(Nb_valtest_by_100-1):
		y_by_100.append(np.mean(y[100*k:100*k+100]))
	y_by_100.append(np.mean(y[5900:5909]))
	

	plt.plot(x, y_train_temp_by_100, "g+", label="Numéro de période de la dernière recharge de la tournée optimale" )
	plt.plot(x, y_by_100, "ro", label="Numéro de période de la dernière recharge de la tournée prédite")
	plt.ylabel('Numéro de période de la dernière recharge')
	plt.xlabel('numéro de l\'instance')
	plt.legend()
	plt.savefig('time_N20_prediction_courbe_Al_He_train.pdf')
	plt.show()


#Construction du réseau de neurones Al_He qui calcule uniquement les valeurs y_i  ici on n'enlève la dernière couche qui calculent i*tau_i
def CustomLoss_uniquement_prod_with_y_i(N, Cv_train, Cf_train, Rend_train, lambda_train, y_train, y_list_int):#, arange_train
	#N = nombre de périodes max

	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  on genere des donnees aleatoirement 

	#Cv_train = []
	#Cf_train = []
	#Rend_train = []
	#lambda_train = []
	#arange_train = []

	#prod_train = []
	#lastRefuel_train = []
	#setup_train = []
	#y_train = []


	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	#on a 4 vecteurs de 5 composantes en entree
	my_input1 = keras.Input(shape=(N,), name="input1") 
	my_input2 = keras.Input(shape=(N,), name="input2")  
	my_input3 = keras.Input(shape=(N,), name="input3") 
	my_input4 = keras.Input(shape=(N+2,), name="input4")  
	##my_input5 = keras.Input(shape=(N, ), name="input5")  #entree fictive qui nous permet de passer des coeff pour faire le prod. scalaire


	#on concatene les input1 et input 2 puis input3 et input4
	concat1 = keras.layers.concatenate([my_input1, my_input2])
	concat2 = keras.layers.concatenate([my_input3, my_input4])

	# on cree deux couches sigmoides de taille N (une par couche concat)
	couche1_A = keras.layers.Dense(N, activation='sigmoid')(concat1)
	couche1_B = keras.layers.Dense(N, activation='sigmoid')(concat2)


	#on concatee les couches precedentes
	concat3 = keras.layers.concatenate([couche1_A, couche1_B])

	#on cree les couches de "présortie"
	out_last_prod = keras.layers.Dense(N, activation='sigmoid', name="out_last_prod")(concat3) #  gamma
	##pre_lastRefuel = keras.layers.Dense(N, activation='softmax', name="out2")(concat3)#  gamma*
	#pre_setup = keras.layers.Dense(N, activation='sigmoid', name="out3")(pre_prod)# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes

	#Ajouter ceci si pour passer de la couche prod a la couche setup il faut un biais nul et des poids (-1,1) fixes et enlever la ligne précédente
	#(poids_setup, biais_setup) = creationPoidsSetup(N)
	#pre_setup = keras.layers.Dense(N, activation='relu', name="out_setup", weights=[poids_setup, biais_setup], trainable=False)(pre_prod) # tau
	

	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	#out_prod = tf.keras.layers.Multiply()([my_input1, pre_prod])
	#out_setup = tf.keras.layers.Multiply()([my_input2, pre_setup])
	##out_lastRefuel = tf.keras.layers.Multiply()([my_input5, pre_lastRefuel])

	#on concatene les couches et on fait la somme de toutes les composantes
	#concat4 = keras.layers.concatenate([out_prod, out_setup])#, out_lastRefuel
	#out_val = keras.layers.Dense(N, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(concat4)



	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2, my_input3, my_input4], outputs=out_last_prod)#, my_input5

	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauClassif.png", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  ajout de la fonction d erreur et apprentissage

	opt = keras.optimizers.SGD(lr=0.015, momentum=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=opt)
	
	#on met les donnees d'entrainement dans le bon format
	#print(Cv_train)

	Cv_train2 = np.asarray(Cv_train[90:]) #Les 90 premières instances seront les instances de test 
	print("debut transforme Cf_train2")
	Cf_train2 = np.asarray(Cf_train[90:])#[90:]
	print("debut transforme Rend_train2")
	Rend_train2 = np.asarray(Rend_train[90:])#[90:]
	print("debut transforme lambda_train2")
	lambda_train2 = np.asarray(lambda_train[90:])#[90:]
	print("milieu transforme data")
	#prod_train = np.asarray(prod_train)
	#lastRefuel_train = np.asarray(lastRefuel_train)
	#setup_train = np.asarray(setup_train)
	##arange_train2 = np.asarray(arange_train[90:])#[90:]
	y_train2 = np.asarray(y_list_int[90:])#[90:]
	print("fin transforme data")

	

	#entrainement
	print (Cv_train2.shape)
	print (type(Cv_train2))
	print (Cf_train2.shape)
	print (type(Cf_train2))
	print (Rend_train2.shape)
	print (type(Rend_train2))
	print (lambda_train2.shape)
	print (type(lambda_train2))
	##print (arange_train2.shape)
	##print (type(arange_train2))
	print (y_train2.shape)
	print (type(y_train2))
	print("debut fit")
	history = model.fit({"input1": Cv_train2, "input2": Cf_train2, "input3":Rend_train2, "input4":lambda_train2},
		  y_train2,
		 epochs=200, batch_size=16, verbose=0)#, "input5": arange_train2
	print("fin fit")

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.title("erreur en fonction des epochs")
	plt.savefig('loss_prediction_courbe_Al_He_.png')
	plt.show()
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 5.  on teste

	print (Cv_train2.shape)
	print (type(Cv_train2))

	print (Cv_train2[0:1].shape)
	print (type(Cv_train2[0:1]))

	y_predict = model.predict({"input1" : np.asarray(Cv_train[0:90]) , "input2":  np.asarray(Cf_train[0:90]),
				   "input3": np.asarray(Rend_train[0:90]), "input4": np.asarray(lambda_train[0:90])})#,"input5": np.asarray(arange_train[0:90])
	#print ("predit : ", y_predict)
	#print("theorique", y_train[0:90])
	Nb_valtest=90
	
	y_predictVal = []
	cpt=0
	for v in y_predict:
		CV=Cv_train[cpt]
		y_predVal=0
		for k in range(len(v)):
				y_predVal=y_predVal+(np.asarray(CV[k])*v[k])
		y_predictVal.append(y_predVal)
		cpt=cpt+1
	#représentation graphique en une unique courbe
	y_train_temp=y_train[0:90]
	# on tri suivant y_train
	print(y_train_temp)
	print(y_predictVal)
	y_train_temp, y_predictVal =zip(*sorted(zip(y_train_temp, y_predictVal)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g", label="Opt" )
	plt.plot(x, y_predictVal, "r", label="Predict")
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet.png')
	plt.show()


def CustomLoss_uniquement_last_recharge_with_tau_i(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train, Dates_last):

	#créons pour chaque instance dont on connait la date de dernière recharge le vecteur tau
	Date_last_recharg=[]
	for dat in Dates_last[90:]:#Les 90 premières instances seront les instances de test 
		list_last_recarh_one=[0]*N
		list_last_recarh_one[dat]=1
		Date_last_recharg.append(list_last_recarh_one)
	#print(len(Dates_last))
	#print(len(Dates_last[90:]))
	#print(len(Date_last_recharg))

	#Date_last_recharg contient la date de dernière recharge pour chaque instance
	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation des couches


	#contrairement au reseau sequential on doit commencer par definir explicitement les couches d'entree
	#on a 4 vecteurs de 5 composantes en entree
	my_input1 = keras.Input(shape=(N,), name="input1") 
	my_input2 = keras.Input(shape=(N,), name="input2")  
	my_input3 = keras.Input(shape=(N,), name="input3") 
	my_input4 = keras.Input(shape=(N+2,), name="input4")  
#my_input5 = keras.Input(shape=(N, ), name="input5")  #entree fictive qui nous permet de passer des coeff pour faire le prod. scalaire


	#on concatene les input1 et input 2 puis input3 et input4
	concat1 = keras.layers.concatenate([my_input1, my_input2])
	concat2 = keras.layers.concatenate([my_input3, my_input4])

	# on cree deux couches sigmoides de taille N (une par couche concat)
	couche1_A = keras.layers.Dense(N, activation='sigmoid')(concat1)
	couche1_B = keras.layers.Dense(N, activation='sigmoid')(concat2)


	#on concatee les couches precedentes
	concat3 = keras.layers.concatenate([couche1_A, couche1_B])

	#on cree les couches de "présortie"
	##pre_prod = keras.layers.Dense(N, activation='sigmoid', name="out1")(concat3) #  gamma
	pre_lastRefuel = keras.layers.Dense(N, activation='softmax', name="out_final")(concat3)#  gamma*
	#pre_setup = keras.layers.Dense(N, activation='sigmoid', name="out3")(pre_prod)# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes

	#Ajouter ceci si pour passer de la couche prod a la couche setup il faut un biais nul et des poids (-1,1) fixes et enlever la ligne précédente
	##(poids_setup, biais_setup) = creationPoidsSetup(N)
	##pre_setup = keras.layers.Dense(N, activation='relu', name="out_setup", weights=[poids_setup, biais_setup], trainable=False)(pre_prod) # tau
	

	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	##out_prod = tf.keras.layers.Multiply()([my_input1, pre_prod])
	##out_setup = tf.keras.layers.Multiply()([my_input2, pre_setup])
	#out_lastRefuel = tf.keras.layers.Multiply()([my_input5, pre_lastRefuel])

	#on concatene les couches et on fait la somme de toutes les composantes
	##concat4 = keras.layers.concatenate([out_prod, out_setup, out_lastRefuel])
	#out_val = keras.layers.Dense(1, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(out_lastRefuel)



	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2, my_input3, my_input4], outputs=pre_lastRefuel)#, my_input5

	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauCustomLoss.png", show_shapes=True, dpi=192)
	

	#---------------------------------------------------------------------------------------------------
	# ETAPE 4.  ajout de la fonction d erreur et apprentissage

	opt = keras.optimizers.SGD(lr=0.015, momentum=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=opt)
	#on met les donnees d'entrainement dans le bon format
	#print(Cv_train)

	Cv_train2 = np.asarray(Cv_train[90:]) #Les 90 premières instances seront les instances de test 
	print("debut transforme Cf_train2")
	Cf_train2 = np.asarray(Cf_train[90:])#[90:]
	print("debut transforme Rend_train2")
	Rend_train2 = np.asarray(Rend_train[90:])#[90:]
	print("debut transforme lambda_train2")
	lambda_train2 = np.asarray(lambda_train[90:])#[90:]
	print("milieu transforme data")
	#prod_train = np.asarray(prod_train)
	#lastRefuel_train = np.asarray(lastRefuel_train)
	#setup_train = np.asarray(setup_train)
	arange_train2 = np.asarray(arange_train)#[90:]
	y_train2 = np.asarray(Date_last_recharg)
	print("fin transforme data")

	

	#entrainement
	print (Cv_train2.shape)
	print (type(Cv_train2))
	print (Cf_train2.shape)
	print (type(Cf_train2))
	print (Rend_train2.shape)
	print (type(Rend_train2))
	print (lambda_train2.shape)
	print (type(lambda_train2))
	print (arange_train2.shape)
	print (type(arange_train2))
	print (y_train2.shape)
	print (type(y_train2))
	print("debut fit")
	history = model.fit({"input1": Cv_train2, "input2": Cf_train2, "input3":Rend_train2, "input4":lambda_train2},
		  y_train2,
		 epochs=200, batch_size=16, verbose=0)
	print("fin fit")#, "input5": arange_train2

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.title("erreur en fonction des epochs")
	plt.savefig('loss_prediction_courbe_Al_He_.png')
	plt.show()
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 5.  on teste

	print (Cv_train2.shape)
	print (type(Cv_train2))

	print (Cv_train2[0:1].shape)
	print (type(Cv_train2[0:1]))

	y_predict = model.predict({"input1" : np.asarray(Cv_train[0:90]) , "input2":  np.asarray(Cf_train[0:90]),
				   "input3": np.asarray(Rend_train[0:90]), "input4": np.asarray(lambda_train[0:90])})

	Nb_valtest=90#Pour toutes les instances c'est 2549

	#On calcule pour chaque instance la valeur prédite pour la date de dernière recharge à l'aide du vecteur tau
	y_predictVal = []
	for v in y_predict:
		y_predictVal.append(calculCout(v,N))

	# on tri suivant y_train
	#représentation graphique en une unique courbe
	y_train_temp=y_train[0:90]
	print("Le vecteur des valeurs optimales de la date de dernière recharge est \n")
	print(y_train_temp)
	y_train_temp, y_predictVal =zip(*sorted(zip(y_train_temp, y_predictVal)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g", label="Opt" )
	plt.plot(x, y_predictVal, "r", label="Predict")
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet.png')
	plt.show()

#Foncction qui prend une liste en entrée et renvoie sa concaténation selon H
def concat_with_H(Nb_neuron_couche_sortie,Nb_element_couch_entree_min,Nb_element_couch_entree_max, H,coucheInput1_A):
	c_list_a_concat_all=[]
	for j in range(Nb_neuron_couche_sortie):
		list_a_concat=[]#liste contenant un groupe de neurones
		for i in range(0,Nb_element_couch_entree_max-Nb_element_couch_entree_min):
			if(abs(i-j)<H):
				list_a_concat.append(coucheInput1_A[i])
		if(len(list_a_concat)>1):
			c_list_a_concat = keras.layers.concatenate(list_a_concat)
			c_list_a_concat_all.append(c_list_a_concat)
		else:
			c_list_a_concat_all.append(list_a_concat)
	return c_list_a_concat_all

def Creation_couche_neurones_par_neurones(H, N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train):
	print(H)
	
	
	#---------------------------------------------------------------------------------------------------
	# ETAPE 1.  creation des couches

	#1.1. on va definir chaque neurones de la couche input separement, on les stocke dans un tableau
	coucheInput = []

	for i in range(5*N+2):
		input_tmp = keras.Input(shape=(1,), name="input"+str(i)) 
		coucheInput.append(input_tmp)

	
	couche1_A = []
	couche1_B = []
	c_list_a_concat_all=[]

#1.2. on va definir chaque neurones de la couche 1 separement, on les stocke dans un tableau
	coucheInput1_A=coucheInput[:2*N]
	c_list_a_concat_all=concat_with_H(N,0,2*N, H,coucheInput1_A)
	for j in range(N):
		tmp = keras.layers.Dense(1, activation='sigmoid')(c_list_a_concat_all[j])
		couche1_A.append(tmp)

	c_list_a_concat_all=[]
	coucheInput1_B=coucheInput[2*N:4*N+2]
	c_list_a_concat_all=concat_with_H(N,2*N,4*N+2, H,coucheInput1_B)
	for j in range(N):
		tmp = keras.layers.Dense(1, activation='sigmoid')(c_list_a_concat_all[j])
		couche1_B.append(tmp)
	
	#on concatee les couches precedentes
	concat3 = couche1_A + couche1_B


	#on cree les couches de "présortie"
	c_list_a_concat_all=[]
	pre_prod=[]
	c_list_a_concat_all=concat_with_H(N,0,2*N, H,concat3)
	for j in range(N):
		tmp = keras.layers.Dense(1, activation='sigmoid')(c_list_a_concat_all[j])
		pre_prod.append(tmp)

	c_list_a_concat_all=[]
	pre_lastRefuel=[]
	c_list_a_concat_all=concat_with_H(N,0,2*N, H,concat3)
	for j in range(N):
		tmp = keras.layers.Dense(1, activation='softmax')(c_list_a_concat_all[j])
		pre_lastRefuel.append(tmp)
	
	# tau #ajouter cette ligne si pour passer de la couche prod a la couche setup il ne faut un biais nul et des poids (-1,1) fixes
	c_list_a_concat_all=[]
	pre_setup=[]
	c_list_a_concat_all=concat_with_H(N,0,N, H,pre_prod)
	for j in range(N):
		tmp = keras.layers.Dense(1, activation='softmax')(c_list_a_concat_all[j])
		pre_setup.append(tmp)


	#on ajoute les coeff aux couches de sortie : on utilise une couche Multiply
	out_prod=[]
	out_setup=[]
	out_lastRefuel=[]
	for j in range(N):
		out_prod.append( tf.keras.layers.Multiply()([coucheInput[j], pre_prod[j]]))
		out_setup.append( tf.keras.layers.Multiply()([coucheInput[j+N], pre_setup[j]]))
		out_lastRefuel.append( tf.keras.layers.Multiply()([coucheInput[j+4*N+2], pre_lastRefuel[j]]))

	out_list=[]
	out_list=out_prod+out_setup+out_lastRefuel
	#print(out_list)
	concat4 = keras.layers.concatenate(out_list)
	out_val = keras.layers.Dense(1, activation='linear', name="out_val", kernel_initializer=tf.keras.initializers.Ones(), trainable=False)(concat4)


	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=coucheInput, outputs=out_val)
	
	#résumé du modèle
	model.summary()

	#dessin du modele
	keras.utils.plot_model(model, "reseauCustomArc.png", show_shapes=True, dpi=192)
	
	# compile model
	print("debut compile")
	model.compile(optimizer='adam', loss='mean_squared_error')

	my_input_pred=[]
	y_train2 = np.asarray(y_train[90:])##

	for j in range(90,y_train2.shape[0]+90):##données d'entrainements sans les données de test 90,y_train2.shape[0]+90.............0,y_train2.shape[0]
		myinput_ligne=Cv_train[j]+Cf_train[j]+Rend_train[j]+lambda_train[j]+arange_train[j]
		my_input_pred.append(myinput_ligne)
	my_input = np.asarray(my_input_pred)
	print(my_input.shape)
	print(y_train2.shape)

	input=[]
	for i in range(5*N+2):
		input.append(my_input[:,i])
	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  entrainement du modele

	#entrainement
	history = model.fit(input, y_train2, epochs=200, batch_size=32, verbose=0)
	

	# on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
	plt.plot(history.history["loss"])
	plt.title("erreur en fonction des epochs")
	plt.savefig('loss_prediction_courbe_Al_He_.png')
	plt.show()

	Nb_valtest=y_train2.shape[0]
	my_input_test = []
	for j in range(0,Nb_valtest):
		myinput_ligne=Cv_train[j]+Cf_train[j]+Rend_train[j]+lambda_train[j]+arange_train[j]
		my_input_test.append(myinput_ligne)
	my_input_test_pred = np.asarray(my_input_test)
	
	input_tes=[]
	for i in range(5*N+2):
		input_tes.append(my_input_test_pred[:,i])

	y = model.predict(input_tes)
	print ("predit : ", y)
	print("theorique", y_train[0:90])
	

	#représentation graphique en une unique courbe
	y_train_temp=y_train
	y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
	x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
	plt.plot(x, y_train_temp, "g", label="Opt" )
	plt.plot(x, y, "r", label="Predict")
	plt.legend()
	plt.savefig('prediction_courbe_Al_He_complet.png')
	plt.show()


	#à supprimer
def reseauFunctional(X_train1, X_train2, y_train, X_train1_shape, X_train2_shape, X_test1, X_test2, y, Nombre_neurones):
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
	couche1_m1 = keras.layers.Dense(Nombre_neurones, activation='sigmoid')(my_input1)
	couche2_m1 = keras.layers.Dense(Nombre_neurones, activation='sigmoid')(couche1_m1) 
	

	#defintion des couches dans le second sous-reseau 
	couche1_m2 = keras.layers.Dense(Nombre_neurones, activation='sigmoid')(my_input2)
	couche2_m2 = keras.layers.Dense(Nombre_neurones, activation='sigmoid')(couche1_m2)

	# on concatene les couches finales des 2 sous-reseaux
	merge1 = keras.layers.concatenate([couche2_m1, couche2_m2])
	merge2 = keras.layers.concatenate([couche2_m1, couche2_m2])

	#on finit avec une couche avec plusieurs neurones
	couche3_m1 = keras.layers.Dense(Nombre_neurones, activation='sigmoid')(merge1)
	couchefinal1 = keras.layers.Dense(Nombre_neurones, activation='sigmoid')(couche3_m1)
	couchefinal2 = keras.layers.Dense(Nombre_neurones, activation='softmax')(merge2)
	
	couchefinal = keras.layers.concatenate([couchefinal1, couchefinal2])
	#---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
	
	model = keras.Model(inputs=[my_input1, my_input2], outputs=couchefinal)

	#dessin du modele
	model.summary()
	keras.utils.plot_model(model, "Al_He_reseauFunctional.png", show_shapes=True, dpi=192)

	#---------------------------------------------------------------------------------------------------
	# ETAPE 3.  ajout de la fonction d erreur et apprentissage


	model.compile(optimizer='adam', loss='my_loss_fn')

	
