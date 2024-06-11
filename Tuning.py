import tensorflow.keras as keras
import numpy as np
import sklearn.model_selection
import os

from math import *

#==================================================================================================
#                           
#-----------------------------------------------------------------------------------------------
# ce fichier teste le tuning d'un model en utilisant scikit-learn
# on utilise ici sur un reseau sequential 
# Ref : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# ce fichier teste le tuning d'un model en utilisant scikit-learn
# on l'utilise ici sur un reseau sequential 
# on va tester ici plus de parametres (en plusieurs étapes)
# a chaque etape on verifie qu'on ameliore le modele
#==================================================================================================

#-----------------------------------------------------------------------------------------------
# ETAPE 1 : il faut faire une fonction qui cree le modele que l'on souhaite tuner et qui le retourne
# cette fonction peut prendre des arguments en parametres, ils doivent tous avoir une valeur par defaut

# nCouche = nb couche cachees sans compter la couches d'entree (= data) et la couche de sortie
# nNeurones = nombre de neurones sur chaque couche
def createModel(dropOutRate= 0.0 , nNeurones = (8,4), maxNorm = 10, activation_name  = 'relu' , optimizer_name   ='adam', loss_name='mean_squared_error', Nb_composantes_input=325) : #  

	#NB. maxNorm est utile pour le dropout, si on utilise un dropout rate = 0, on peut supprimer le maxnorm
	
	#print ("creation modele avec dropOut = ", dropOutRate, "nNeurones = ", nNeurones, "maxNorm = ", maxNorm)

	#si nNeurones = (16,10,20) : alors je crée une couche cachee avec 16 neurones, 
	#puis une seconde couche cachee avec 10 neurones puis une dernière couche cachee avec 20 neurones


	# on construit un reseau sequantial
	model = keras.Sequential()

	# premiere couche cachee
	model.add(keras.layers.Dense(nNeurones[0], activation=activation_name, 
							  #kernel_constraint=keras.constraints.max_norm(maxNorm), 
							  input_shape=(Nb_composantes_input,)))
	model.add(keras.layers.Dropout(dropOutRate)) 

	# les autres : 
	
	for i in range(len(nNeurones)-1):
		#print("n neurone couche", i+1, " = ", nNeurones[i+1])
		model.add(keras.layers.Dense(nNeurones[i+1], activation=activation_name, ))#kernel_constraint=keras.constraints.max_norm(maxNorm)))
		model.add(keras.layers.Dropout(dropOutRate)) 
	
	# derniere couche : un seul neurone qui donne la solution
	model.add(keras.layers.Dense(1))


	# on definit la fonction d'erreur
	model.compile(optimizer=optimizer_name, loss=loss_name)

	# on retourne le model
	return model


#-----------------------------------------------------------------------------------------------
# ETAPE 2 : on passe la fonction de creation du model a l'argument build_fn de la fonction 
# KerasRegressor, on peut ajouter les arguments de la fonction fit (epochs par exemple) et les
# arguments de la fonction  create_model (ici dropOutRate)


# on commence par definir la fonction d'erreur que l'on souhaite utiliser
# ici un gap entre la solution attendue (y_true) et la solution predite (y_pred)
# ATTENTION : pour utiliser cette metrique on doit etre sur que y_true est toujours != 0

def mean_absolute_percentage_error(y_true, y_pred): 
    
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	#on transforme y_pred qui est a priori une matrice a une colonne en vecteur unidimensionel
	y_pred = np.asarray(y_pred).reshape(-1)

	if y_pred.shape != y_true.shape:
		print ("mean_absolute_percentage_error : erreur dimension non homogene")
		exit(-1)
		
	diff = abs(y_true - y_pred) * 100
	return np.mean(diff / y_true)



def tuning(X, y, nbParallelJob,Nb_composantes_input):

	# 1. on construit le model via KerasRegressor
	model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=createModel, verbose=0)

	# 2. on definit via un dictionnaire tous les parametres que l on souhaite tester
	if os.name == 'nt' : #windows
		print ("on fait le tuning sous windows")
		paramGrid = dict(dropOutRate=[0.0], 
				   nNeurones = [(8,4)], #, (16, 8, 4)
				  # maxNorm = [10],
				   activation_name = ['relu'],
				   optimizer_name = ['adam'], 
				   loss_name = ['mean_squared_error'], 
				   epochs=[50,100, 200],
				   batch_size=[16,32,40,52],
				   Nb_composantes_input=[Nb_composantes_input])
	else:
		print ("on fait le tuning sous noyau linux")
		paramGrid = dict(dropOutRate=[0.0],#, 0.2, 0.4], 
				   nNeurones = [(8, 4), (16, 8, 4)],#, (32, 16, 4)], 
				   #maxNorm = [3, 5, 10, 50],
				   activation_name = ['relu'],
				   optimizer_name = ['adam'], 
				   loss_name = ['mean_squared_error'], 
				   epochs=[400, 600, 800], 
				   batch_size=[4, 16, 32],
				   Nb_composantes_input=[Nb_composantes_input])

	# 3. on lance la recherche des meilleurs param avec GridSearchCV :
	# cette fonction teste toutes les combinaisons de parametres possible  
	# utilise la K-fold Cross Validation avec cv = 3 k-fold par defaut (5 semble meilleur)
	#  Cross Validation : on divise les donnees en K paquets (K-1 serviront a l'apprentissage et 1 au test) et on  teste K fois (le paquet test change a chaque fois)
	# on peut fixer le nb de thread pour la recherche avec n_jobs (-1 = tous les CPU dispo)

	# make_scorer : permet de definir une fonction de scoring pour GridSearchCV
	score_percentage_error = sklearn.metrics.make_scorer(mean_absolute_percentage_error, greater_is_better=False)

	#remarque : GridSearchCV suppose que le plus grand score est meilleur, c'est pourquoi dans le cas ou c 'est le contraire, il retourne un nb negatif
	grid = sklearn.model_selection.GridSearchCV(estimator=model, scoring=score_percentage_error, param_grid=paramGrid, n_jobs=nbParallelJob , cv=5)





	# 4. on recupere le resultat avec grid.fit
	# best_score_ member provides access to the best score observed during the optimization procedure 
	# best_params_ describes the combination of parameters that achieved the best results.
	grid_result = grid.fit(X, y)

	# 5. affichage du resultat 
	#Best score is the largest mean score across the CV folds for the corresponding configration.
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	print("\n")
	print("\n")
	print(grid_result)
	print("\n")
	print("\n")
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

	return grid_result.best_params_





# fonction principale : genere des donnees, tune le modele via la fonction de tuning definie precedemment
# reconstruit le modele avec les meilleurs param trouves par le tuning, puis teste et affiche les resultats
def modeleTune(nbParallelJob,X,y,Xtest, ytest, Max_tail_list):

	# 0. min, max des donnees pour l apprentissage
	print ("max(y) = ", max(y))
	print ("min(y) = ", min(y))

	# 1. on lance le tuning et on recupere les meilleurs parametres
	bestParam = tuning(X, y, nbParallelJob, Max_tail_list)
	print(bestParam)
	print ("verification : on recree le modele avec les meilleurs param")

	# 2. on cree le modele avec les meilleurs parametres
	model = createModel(dropOutRate=bestParam['dropOutRate'], nNeurones=bestParam['nNeurones'], 
					 #maxNorm=bestParam['maxNorm'],
					 activation_name= bestParam['activation_name'], optimizer_name = bestParam['optimizer_name'], loss_name= bestParam['loss_name'], Nb_composantes_input= Max_tail_list)

	
	# 3. on realise l'apprentissage avec les meilleurs param
	model.fit(X, y, epochs=bestParam['epochs'], batch_size=bestParam['batch_size'], verbose=0)

	
	
	## 2. on cree le modele avec les meilleurs parametres
	#model = createModel()

	## 3. on realise l'apprentissage avec les meilleurs param
	#model.fit(X, y, epochs=100, batch_size=64, verbose=0)

	keras.utils.plot_model(model, 'modelTune.png', show_shapes=True)


	# 4. on teste sur le jeu de donnees de test
	yp = model.predict(Xtest)

	N = len(Xtest) # nb de vecteurs a tester
	err = 0
	for i in range(N):
		err = err + (abs(ytest[i]-yp[i]) * 100 / ytest[i])

	err = err/N
	print ("ERREUR (%) = ", err)
	print ("scoring : ", mean_absolute_percentage_error(ytest, yp))

	

