from sklearn.model_selection import train_test_split
import Read_Instances as RI # Contient les fonctions qui permettent de lire les fichiers contenant les instances
import Sequential_Model as SM
import tensorflow as tf
import Tuning as KTA
import Functional_model as FM
import numpy as np
import Analyse_PCA_data as PCA
import sys
#import Al_He_Functional_model_production as FMP
import Al_He_Functional_model_production as CL
from collections import Counter
import matplotlib.pyplot as plt
import Alternative_learning_scheme as AFMP
#min_num_inst = 70
#max_num_inst = 2621#debug 200
min_num_inst = 10000##
max_num_inst = 15999##
#nombre_d_instance =  max_num_inst - min_num_inst
#========================================================
# I.
#========================================================

#Lecture des instances
#Test fonction
#lines = RI.Read_Sequential("InstanceReseauNeurones/RN_instance_prod__", min_num_inst)
#rend, CoutFixe, CoutVar, CoutProd, DateDebLastRecharg, QteRecharg = RI.Read_Line_By_Line("InstanceReseauNeurones/RN_instance_prod__", 70)
#x_data, y_data = RI.Input_Output_RN_By_Type("InstanceReseauNeurones/RN_instance_prod__", min_num_inst)
#x_data_p, y_data_p = RI.Input_Output_RN_Period_By_Period("InstanceReseauNeurones/RN_instance_prod__", min_num_inst)
X_data, Y_data= RI.X_data_Y_data_construct(min_num_inst, max_num_inst, 1,"InstanceReseauNeurones/RN_instance_prod__")#en mettant 1 ou 0 cela permet de classé
                                                                                                                     #les données par type ou par période
#print(min(Y_data))
#print(max(Y_data))
#calcul de la taille de la plus grande liste de X_data
Max_tail_list=0
for i in range(0, len(X_data)):
    if(len(X_data[i]) > Max_tail_list):
        Max_tail_list=len(X_data[i])

#on transforme chaque liste de X_data en liste de taille Max_tail_list en completant la liste pas des 0
for i in range(0, len(X_data)):
    if(len(X_data[i]) < Max_tail_list):
        list_0 = [0 for i in range(Max_tail_list-len(X_data[i]))]
        X_data[i].extend(list_0)
X_train_me=np.asarray(X_data[90:])
y_train_me=np.asarray(Y_data[90:])
X_test_me=np.asarray(X_data[0:90])
y_test_me=np.asarray(Y_data[0:90])
##diviser en données de test et données d'entrainement
#X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

#X_train_and_test = X_train+X_test
#y_train_test=y_train+y_test
#X_train=X_train_and_test[90:]
#X_test=X_train_and_test[0:90]
#y_train=y_train_test[90:]
#y_test=y_train_test[0:90]
##transformer la liste de listes en tenseurs
##Y_data_tensor = tf.convert_to_tensor(y_train)
##X_data_tensor = tf.convert_to_tensor(X_train)
##Y_data_test_tensor = tf.convert_to_tensor(y_test)
#X_data_test_tensor = tf.convert_to_tensor(X_test)
#X_data_train_tensor = tf.convert_to_tensor(X_train)
#========================================================
# II.
#========================================================

# création d'un modèle sans tuning (les parties qui sont commentées #...# sont à décommenter si on veut les exécuter
#construction du modèle RN
model = SM.MLP_LN_regression_construct(Max_tail_list)#

#compilation et entrainement du modèle RN
SM.MLP_LN_regression_FctError_Apprentissag_SaveModel(model, 'adam', 'mean_squared_error', X_train_me, y_train_me, 200, 32, 0)#
#print(X_data)
#print(Y_data)

#prediction

#test sur les instances d'apprentissage
SM.MLP_LN_regression_predire(len(X_train_me), X_train_me, y_train_me,'SequentialModel_prediction_train.pdf')#


#test sur les instances de test
SM.MLP_LN_regression_predire(len(X_test_me), X_test_me, y_test_me, 'SequentialModel_prediction_test.pdf')#

##========================================================
# III.
#========================================================

# tuning
#nbParallelJob = 1#

#if len(sys.argv) == 2:#
	#nbParallelJob = int(sys.argv[1])#
	#print ("utilisation de ", nbParallelJob, " jobs en parallel")#
#else:#
	#print ("un seul thread")#

#KTA.modeleTune(nbParallelJob, X_train, y_train,X_test, y_test, Max_tail_list) #

#========================================================
# IV.
#========================================================

# exemple de construction d'un reseau functional 

#X_train1 = X_train[:len(X_train)//2] #
#X_train2 = X_train[len(X_train)//2:len(X_train)]#
#type(X_train)#
#X_train_nparray=np.array(X_train)#
#type(X_train_nparray)#
#X_train1 = X_train_nparray[:,:Max_tail_list//2] #
#X_train2 = X_train_nparray[:,Max_tail_list//2:Max_tail_list]#
#print(X_train1.shape)#
#print(X_train2.shape)#
#print(X_train_nparray.shape)
#X_train1_shape = len(X_train1[0])#Nombre de colonnes des donnees#
#X_train2_shape = len(X_train2[0])##
#print(X_train1_shape)#
#print(X_train2_shape)#
#X_test_nparray=np.array(X_test)#
#X_test1 =  X_test_nparray[:,:Max_tail_list//2]##
#X_test2 = X_test_nparray[:,Max_tail_list//2:Max_tail_list]##
#print(len(X_test1))#
#print(len(X_test2))#
#print(len(X_test))#
#y = y_test#
#FM.reseauFunctional(X_train1, X_train2, y_train, X_train1_shape, X_train2_shape, X_test1, X_test2, y)#

#========================================================
# V. 
#========================================================

#Analyse PCA des donnees
#X_data_nparray=np.array(X_data)#
#nombre d'observations
#n = X_data_nparray.shape[0]#
#  #nombre de variables 
#p = X_data_nparray.shape[1]#
#Z=PCA.preparation_data(X_data_nparray)#
#PCA.analyse_PCA(X_data_nparray,Z,n)#


#========================================================
# VI. 
#========================================================

#statistique des inputs

#RI.Input_Output_statistiques("instances_avec_p/instance__",max_num_inst, min_num_inst)#
#========================================================
# VII. 
#========================================================

#Reseau fonctionnel Al_He


#X_train1 = X_train[:len(X_train)//2] #
#X_train2 = X_train[len(X_train)//2:len(X_train)]#
#type(X_train)#
#X_train_nparray=np.array(X_train)#
#type(X_train_nparray)#
#X_train1 = X_train_nparray[:,:Max_tail_list//2] #
#X_train2 = X_train_nparray[:,Max_tail_list//2:Max_tail_list]#
#print(X_train1.shape)#
#print(X_train2.shape)#
#print(X_train_nparray.shape)
#X_train1_shape = len(X_train1[0])#Nombre de colonnes des donnees#
#X_train2_shape = len(X_train2[0])##
#print(X_train1_shape)#
#print(X_train2_shape)#
#X_test_nparray=np.array(X_test)#
#X_test1 =  X_test_nparray[:,:Max_tail_list//2]##
#X_test2 = X_test_nparray[:,Max_tail_list//2:Max_tail_list]##
#print(len(X_test1))#
#print(len(X_test2))#
#print(len(X_test))#
#y = y_test#
#CL.reseauFunctional(X_train1, X_train2, y_train, X_train1_shape, X_train2_shape, X_test1, X_test2, y,4)#

#========================================================
# VII. 
#========================================================

#========================================================

#Reseau fonctionnel Al_He



#Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train, N, y_list_int, Nb_recharge, Date_last_recharg, kkk = RI.Input_Output_CustomLoss("instances_avec_p/instance__", "InstanceReseauNeurones/RN_instance_prod__","InstanceReseauNeurones_suite/RN_instance_prod_suite__", min_num_inst, max_num_inst,0)#CustomLoss ou CustomLoss_with_drop_out
#Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train, N, y_list_int, Nb_recharge, Date_last_recharg,kkk = RI.Input_Output_CustomLoss("instances_avec_p/instance__", "InstanceReseauNeurones/RN_instance_prod__","InstanceReseauNeurones_suite/RN_instance_prod_suite__", min_num_inst, max_num_inst,2)#CustomLoss_uniquement_prod
#Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train, N, y_list_int, Nb_recharge, Date_last_recharg,kkk = RI.Input_Output_CustomLoss("instances_avec_p/instance__", "InstanceReseauNeurones/RN_instance_prod__","InstanceReseauNeurones_suite/RN_instance_prod_suite__", min_num_inst, max_num_inst,1)#CustomLoss_uniquement_last_recharge
#Si le dernier élément de la fonction Input_Output_CustomLoss vaut 0 alors y_train=last_refuel+coutprod
#Si le dernier élément de la fonction Input_Output_CustomLoss vaut 1 alors y_train=last_refuel
#Si le dernier élément de la fonction Input_Output_CustomLoss vaut 2 alors y_train=coutprod
    

#on transforme chaque liste en liste de taille N en completant la liste pas des N-len(Cv_train[i]) de la liste
#exemple : 1 2 3 4 1 2 3

#Cv_train=RI.Transforme_list__same_tail(Cv_train,N)#
#print(len(Cv_train[1]))#
#Cf_train=RI.Transforme_list__same_tail(Cf_train,N)#
#print(len(Cf_train[1]))#
#Rend_train=RI.Transforme_list__same_tail(Rend_train,N)#
#print(len(Rend_train[1]))#
#lambda_train=RI.Transforme_list__same_tail(lambda_train,N+2)#
#a= len(lambda_train[0])#
# #print(a)
#for i in lambda_train:#
#    if(len(i)!=a):#
#        break#
        #print("Problème")#
        #print(len(i))#
#        arange_train=RI.Transforme_list__same_tail(arange_train,N)#
#print(len(arange_train[1]))#
#y_list_int=RI.Transforme_list__same_tail(y_list_int,N)#


#diviser en données de test et données d'entrainement
#print("customloss\n")#

#entrainer le modèle
#CL.CustomLoss(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train)#

#Entrainement du modèle avec drop out 0.2
#CL.CustomLoss_with_drop_out(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train, 0.5)

#Entrainement du modèle qui calcule uniquement le cout de production en utilisant les valeurs y_i et y_i*
#CL.CustomLoss_uniquement_prod(N, Cv_train, Cf_train, Rend_train, lambda_train, y_train,kkk)# arange_train,

#Entrainement du modèle qui calcule uniquement le cout véhicule (date de dernière recharge) en utilisant les valeurs i*tau_i
#CL.CustomLoss_uniquement_last_recharge(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train)

#Construction du réseau de neurones Al_He qui calcule uniquement les valeurs y_i  ici on n'enlève la dernière couche qui calculent i*tau_i
#CL.CustomLoss_uniquement_prod_with_y_i(N, Cv_train, Cf_train, Rend_train, lambda_train, y_train, y_list_int)

#Entrainement du modèle qui calcule uniquement tau_i ici on n'enlève la dernière couche qui calculent y_i
#CL.CustomLoss_uniquement_last_recharge_with_tau_i(N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train, Date_last_recharg)
#!!!!!ATTENTION : Vérifier que les inputs Input_Output_CustomLoss sont bien les bonnes inputs


#On va recrée le modèle CustomLoss mais cette fois ci nous n'allons plus relier les neurones avec dense mais comme ceci:
#Si A est la couche qui précède la couche B alors le neurones B_i aura comme neurones d'entrés les neurones A_{i-H}...A_{i+H}
#H=2 #!!!!!ATTENTION : choisir le bon Input_Output_CustomLoss !!!!!!!!!!!!!!!!!!!!!!!!
#CL.Creation_couche_neurones_par_neurones(H, N, Cv_train, Cf_train, Rend_train, lambda_train, arange_train, y_train)

#statistique sur le nombre d'instances et le nombre de recharge
#X=[]#
#Y=[]#
#count_Nb_recharge = Counter(Nb_recharge).most_common()# #compte le nombre d'occurence de chaque entier d'une liste
#nombre d'instances pour chaque M
#for i in range(0,len(count_Nb_recharge)) :#
#    X.append(count_Nb_recharge[i][0])#
#    Y.append(count_Nb_recharge[i][1])#
#for i in range(0,len(X)) :#
#    X[i] = int(X[i])#
 #print(X)
 #print(Y)
#plt.bar(X,Y)#
 #plt.legend()
#plt.xlabel("# recharges")#
#plt.ylabel("# instances")#
#plt.savefig('Nombre_dinstance_par_nb_recharge.png')#
#plt.show()#

#------------------------------------------Alternative learning for H2 production-------------------------------------------------------------------------------------
#mu_ind, K_ind, R_ind, R0_ind, C_ind, C_q_ind, C__ind, I0_ind, R__ind, P__ind, V__ind, A__ind, G__ind, Iq_ind, Delta_q_ind, Delta_ind, P_ind, P0_ind, G_ind, G0_ind, A_ind, A0_ind, V_ind, V0_ind, H0, N, period_last_recharge, Lambda, Min, Max, Q, CoutProd_one, out_cost_A_P_one_pipeline, period_last_recharge_pipeline,TTTT, CMP=AFMP.Indicator_of_one_instance("instances_avec_p/instance__", "InstanceReseauNeurones/RN_instance_prod__","InstanceReseauNeurones_suite/RN_instance_prod_suite__","InstanceReseauNeurones_suite_suite/RN_instance_prod_suite_suite__",94)
#input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge, output, recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta = AFMP.Input_output_Time_value("instances_avec_p/instance__", "InstanceReseauNeurones/RN_instance_prod__","InstanceReseauNeurones_suite/RN_instance_prod_suite__","InstanceReseauNeurones_suite_suite/RN_instance_prod_suite_suite__", min_num_inst, max_num_inst)
#AFMP.Time_value_model(input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge, recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta, output)
#AFMP.Time_value_predict(input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge, recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta, output)

#in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list = AFMP.Input_output_Cost_value("instances_avec_p/instance__", "InstanceReseauNeurones/RN_instance_prod__","InstanceReseauNeurones_suite/RN_instance_prod_suite__","InstanceReseauNeurones_suite_suite/RN_instance_prod_suite_suite__", min_num_inst, max_num_inst)
#AFMP.Cost_value_model(in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list)
#AFMP.Cost_value_predict(in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list)
