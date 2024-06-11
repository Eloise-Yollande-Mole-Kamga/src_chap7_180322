from math import *
import Read_Instances as RI
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
#debug from numpy.random import seed
#debug seed(1)
import tensorflow as tf
from tensorflow.keras.models import load_model
import Al_He_Functional_model_production as CL
#debug tf.random.set_seed(2)

#Fonction qui calcule les indicateurs pour une instance dont le numéro est placé en entré

def Indicator_of_one_instance(chemin_instance, chemin_prodRN, chemin_prodRN_suite,chemin_prodRN_suite_suite, num_inst):
    # les entrées dont don a besoin pour calculer les indicateurs sont :
    #H0= Recharge initiale de la citerne
    #CMP= Capacité maximal de la citerne
    #Mu= Quantité h2 à charger à chaque recharge q
    #Min= date au plutot de la recharge q
    #Max= #date au plutard de la recharge q
    #delta= delai minimal entre les recharges
    #R= rendements de producion
    #A=Couts d'activation
    #V= cout de production
    #P=V/R cout de production unitaire
    #G=P(i+1)-p(i) gap entre de cout consécutifs
    if(num_inst==2619):
        print("------------------")
    cheminComplet_test = "cinq_instances_pr_test.txt"
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(num_inst) + "\n")
    #Lecture du fichier "chemin_instance"
    ch_num_file = "%d" % num_inst # Transforme num_file en chaine de caractere
    cheminComplet = chemin_instance + ch_num_file + ".txt"
    with open(cheminComplet, "r") as fichier_RS:
            M_list=fichier_RS.readline()
            N_list=fichier_RS.readline()#TMAX
            v0_list=fichier_RS.readline()
            vmax_list=fichier_RS.readline()
            p_list=fichier_RS.readline()
            alpha_list=fichier_RS.readline()
            beta_list=fichier_RS.readline()
            cout_fix_list=fichier_RS.readline()
            E0_list=fichier_RS.readline()
            Ctank_list=fichier_RS.readline()
            concat_test="H0="+E0_list+"CMP="+ Ctank_list+"N="+ str(int(N_list)/int(p_list))
            if(int(N_list)/int(p_list)<=15):
                print("stop")
    #Lecture du fichier contenant T,cost_A,cost_P et delta .......................
    ch_num_file = "%d" % num_inst # Transforme num_file en chaine de caractere
    cheminComplet = chemin_prodRN_suite_suite + ch_num_file + ".txt"
    with open(cheminComplet, "r") as fichier_RS_suit:
        T=fichier_RS_suit.readline()
        cost_A=fichier_RS_suit.readline()
        cost_P=fichier_RS_suit.readline()
        deltaa=fichier_RS_suit.readline()
    #Lecture du fichier "chemin_prodRN"
    rend_one, CoutFixe_one, CoutVar_one, CoutProd_one, DateDebLastRecharg_one, QteRecharg_one = RI.Read_Line_By_Line(chemin_prodRN, num_inst)
    period_last_recharge=int(DateDebLastRecharg_one)//int(p_list) #période last recharge pour la solution optimale
    period_last_recharge_pipeline=int(T)//int(p_list)#période last recharge pour la solution heuristique pipeline
    Cv_train = []
    Cf_train = []
    Rend_train = []
    for j in CoutVar_one.split():
            Cv_train.append(int(j))
    for j in CoutFixe_one.split():
            Cf_train.append(int(j))
    for j in rend_one.split():
            Rend_train.append(int(j))
    CoutProd_one=int(CoutProd_one) #cout de production de la solution optimale
    cost_A=int(cost_A) #cout d'activation de la solution heuristique pipeline
    cost_P=int(cost_P) #cout de production de la solution heuristique pipeline
    #Lecture du fichier "chemin_prodRN_suite"
    mm_list_int=[]
    MM_list_int=[]
    muS_list=[]
    mu=[]
    ch_num_file = "%d" % num_inst # Transforme num_file en chaine de caractere
    cheminComplets = chemin_prodRN_suite + ch_num_file + ".txt"
    with open(cheminComplets, "r") as fichier_RS:
            #y_list = fichier_RS.readline() #désacticver ceci pour exécuter les 6000 instances #vecteur production qui correspond à y dans le modèle réseau de neurones
            y_list='0 0 0 0'#acticver ceci pour exécuter les 6000 instances
            #y_list=fichier_RS.readline() #vecteur production qui correspond à y dans le modèle réseau de neurones
            Nb_recharge=fichier_RS.readline() #le nombre de recharge calculé par la partie véhicule du pipeline
            mm_list=fichier_RS.readline() #periode au plutot possible de chaque s ieme recharge
            MM_list=fichier_RS.readline() #periode au plustard possible de chaque s ieme recharge
            muS_list=fichier_RS.readline()#Quantite d'hydrogene recharger par le vehicule a chaque s ieme recharge
    for j in mm_list.split():
            mm_list_int.append(int(j))
    for j in MM_list.split():
            MM_list_int.append(int(j))
    for j in muS_list.split():#remplisage de la liste mu
            mu.append(int(j))
    Nb_recharge=int(Nb_recharge)#conversion en entier de Nb_recharge
    
    #Entrées qui nous serviront à calculer les indicateurs
    H0 = int(E0_list) #Recharge initiale de la citerne
    CMP = int(Ctank_list) #Capacité maximal de la citerne
    Q = Nb_recharge #nombre de recharge
    concat_test=concat_test+"Q="+ str(Q) + "\n"
    with open(cheminComplet_test, "a") as fichier_RSS:
                fichier_RSS.write(concat_test)
                fichier_RSS.write("Min =" + mm_list)
                fichier_RSS.write("Max =" + MM_list)
                fichier_RSS.write("Mu =" + muS_list + "\n")
    Mu = mu #Quantité h2 à charger à chaque recharge q
    Min = mm_list_int #date au plutot de la recharge q
    Max = MM_list_int #date au plutard de la recharge q
    #delta= delai minimal entre les recharges
    N= int(N_list)/int(p_list) #Nombre de périodes
    R= Rend_train #rendements de producion
    A= Cf_train #couts d'activation
    V= Cv_train #cout de production
    #P=V/R cout de production unitaire
    P=[]
    for j in range(0, len(V)):
        P.append(V[j]/R[j])
    #G=P(i+1)-p(i) gap entre de cout consécutifs
    G=[]
    for j in range(0, len(P)-1):
        G.append(abs(P[j+1]-P[j]))
    with open(cheminComplet_test, "a") as fichier_RSS:
               fichier_RSS.write("R =" + str(R) + "\n")
               fichier_RSS.write("A =" + str(A)+ "\n")
               fichier_RSS.write("V =" + str(V)+ "\n")
               fichier_RSS.write("P =" + str(P)+ "\n")
               fichier_RSS.write("G =" + str(G)+ "\n")
#Calcul de Energy indicators
    #mu----------------------------
    som_mu=0
    for j in Mu:
        som_mu = som_mu + j
    mu_ind=som_mu
    #K----------------------------
    K_ind = ceil(som_mu/CMP)
    #R----------------------------
    som_R=0
    for j in R:
        som_R = som_R + j
    R_ind=som_R/N
    #R0----------------------------
    som_Ri_R=0
    for j in R:
        som_Ri_R = som_Ri_R + abs(j-R_ind)
    R0_ind=som_Ri_R/N
    concat_test_E_ind="mu="+str(mu_ind)+ "\n"+"K="+str(K_ind)+ "\n"+"R="+str(R_ind)+ "\n"+"R0="+str(R0_ind)
    with open(cheminComplet_test, "a") as fichier_RSS:
               fichier_RSS.write(concat_test_E_ind + "\n")
#Calcul de out_cost_A_P
    #out_cost_A_P_one_pipeline = K_ind*cost_A+mu_ind*cost_P #cost_A+cost_P est le cout de production de la solution pipeline
    out_cost_A_P_one_pipeline = cost_A+cost_P
#Calcul de Free indicators
    #C----------------------------
    C_ind=som_R/mu_ind
    #C_q----------------------------
    C_q_ind=[]
    for q in range(0,Q):
        som_max_q=0
        for i in range(0, Max[q]):
            som_max_q=som_max_q+R[i]
        som_mu_q_part=0
        for i in range(0,q+1):
            if(i<Q):
                som_mu_q_part=som_mu_q_part+Mu[i]
        som_mu_q_part=som_mu_q_part-H0
        if(som_mu_q_part>0):
            C_q_ind.append(som_max_q/som_mu_q_part)
        else:
            C_q_ind.append(1000000)
    #C*----------------------------
    C__ind = min(C_q_ind)
    if(C__ind<0):
        print("!!!!!!!!!!!!!!!!!!STOP")
    concat_test_F_ind="C="+str(C_ind)+ "\n"+"C*="+str(C__ind)
    with open(cheminComplet_test, "a") as fichier_RSS:
               fichier_RSS.write(concat_test_F_ind + "\n")
               fichier_RSS.write("C(q) =" + str(C_q_ind) + "\n")
#Calcul de Time indicators
    #I0----------------------------
    I0=-1
    som_Ri_part_i0=0
    while(som_Ri_part_i0<som_mu and I0 < N):
        som_Ri_part_i0=0
        I0=I0+1
        for j in range(0,I0+1):
            if(j<N):
                som_Ri_part_i0 = som_Ri_part_i0 + R[j]
                
    I0_ind = I0+1
    #Calcul de toutes les sommes
    som_i_Ri=0
    som_Ri=0
    som_i_Pi=0
    som_Pi=0
    som_i_Vi=0
    som_Vi=0
    som_i_Ai=0
    som_Ai=0
    som_i_Gi=0
    som_Gi=0
    for i in range(0, int(N)):
        som_i_Ri=som_i_Ri+(i+1)*R[i]
        som_Ri=som_Ri+R[i]
        som_i_Pi=som_i_Pi+(i+1)*P[i]
        som_Pi=som_Pi+P[i]
        som_i_Vi=som_i_Vi+(i+1)*V[i]
        som_Vi=som_Vi+V[i]
        som_i_Ai=som_i_Ai+(i+1)*A[i]
        som_Ai=som_Ai+A[i]
    for i in range(0, int(N)-1):
        som_i_Gi = som_i_Gi + (i+1)*G[i]
        som_Gi = som_Gi + G[i]
    #R*----------------------------
    if(som_Ri!=0):
        R__ind=som_i_Ri/som_Ri
    else:
        R__ind=0
    #P*----------------------------
    if(som_Pi!=0):
        P__ind=som_i_Pi/som_Pi
    else:
        P__ind=0
    #V*----------------------------
    if(som_Vi!=0):
        V__ind=som_i_Vi/som_Vi
    else:
        V__ind=0
    #A*----------------------------
    if(som_Ai!=0):
        A__ind=som_i_Ai/som_Ai
    else:
        A__ind=0
    #G*----------------------------
    if(som_Gi!=0):
        G__ind=som_i_Gi/som_Gi
    else:
        G__ind=0
    #Iq----------------------------
    som_mu_q=-H0
    Iq_ind=[]
    q=0
    while(q<Q):
        som_mu_q= som_mu_q+Mu[q]
        I0=0
        som_Ri_part_i0=R[0]
        while(som_Ri_part_i0<som_mu_q and I0 < N):
            I0=I0+1
            som_Ri_part_i0=0
            for j in range(0,I0+1):
                if(j<N):
                    som_Ri_part_i0 = som_Ri_part_i0 + R[j] 
        Iq_ind.append(I0+1)
        q=q+1
    #Delta_q----------------------------
    q=0
    Delta_q_ind=[]
    while(q<Q):
        Delta_q_ind.append(max(Iq_ind[q],Min[q])-Min[q])
        q=q+1
    #Delta----------------------------
    Delta_ind=max(Delta_q_ind)
    concat_test_T_ind="R*="+str(R__ind)+ "\n"+"P*="+str(P__ind)+ "\n"+"V*="+str(V__ind)+ "\n"+"A*="+str(A__ind)+ "\n"+"G*="+str(G__ind)+ "\n"
    with open(cheminComplet_test, "a") as fichier_RSS:
               fichier_RSS.write("I0="+str(I0_ind)+ "\n")
               fichier_RSS.write(concat_test_T_ind)
               fichier_RSS.write("I(q) =" + str(Iq_ind) + "\n")
               fichier_RSS.write("Delta(q) =" + str(Delta_q_ind) + "\n")
               fichier_RSS.write("Delta="+str(Delta_ind)+ "\n")
#Calcul de Unit Cost indicators
    #P----------------------------
    som_P=0
    for j in P:
        som_P = som_P + j
    P_ind=som_P/N
    #P0----------------------------
    som_Pi_P=0
    for j in P:
        som_Pi_P = som_Pi_P + abs(j-P_ind)
    P0_ind=som_Pi_P/N
     #G----------------------------
    som_G=0
    for j in G:
        som_G = som_G + j
    G_ind=som_G/N
    #G0----------------------------
    som_Gi_G=0
    for j in G:
        som_Gi_G = som_Gi_G + abs(j-G_ind)
    G0_ind=som_Gi_G/N
#Calcul de Absolute cost indicator
    #A----------------------------
    som_A=0
    for j in A:
        som_A = som_A + j
    A_ind=som_A/N
    #A0----------------------------
    som_Ai_A=0
    for j in A:
        som_Ai_A = som_Ai_A + abs(j-A_ind)
    A0_ind=som_Ai_A/N
     #V----------------------------
    som_V=0
    for j in V:
        som_V = som_V + j
    V_ind=som_V/N
    #V0----------------------------
    som_Vi_V=0
    for j in V:
        som_Vi_V = som_Vi_V + abs(j-V_ind)
    V0_ind=som_Vi_V/N
    concat_test_U_A_ind="P="+str(P_ind)+ "\n"+"P0="+str(P0_ind)+ "\n"+"G="+str(G_ind)+ "\n"+"G0="+str(G0_ind)+ "\n"+"A="+str(A_ind)+ "\n"+"A0="+str(A0_ind)+ "\n"+"V="+str(V_ind)+ "\n"+"V0="+str(V0_ind)+ "\n"
    with open(cheminComplet_test, "a") as fichier_RSS:
               fichier_RSS.write(concat_test_U_A_ind)
    if((Min[Q-1] + Delta_ind- Max[Q-1])!=0):
        Lambda=(period_last_recharge_pipeline - Max[Q-1])/(Min[Q-1] + Delta_ind- Max[Q-1]) #period_last_recharge_pipeline
        if(Lambda<0 or Lambda>1 or period_last_recharge_pipeline<Min[Q-1] + Delta_ind or int(T)==1000000):#period_last_recharge_pipeline
            cheminComplet = "PB_INSTANCE.txt"#meeeee
            with open(cheminComplet, "a") as fichier_RS:
                fichier_RS.write(str(num_inst))
                fichier_RS.write("\n")
            print(num_inst)
            #print("!!!!!!!!!!!!!!!!STOP!!!!!!!!!!!!!")
    else:
        Lambda=0
    return mu_ind, K_ind, R_ind, R0_ind, C_ind, C_q_ind, C__ind, I0_ind, R__ind, P__ind, V__ind, A__ind, G__ind, Iq_ind, Delta_q_ind, Delta_ind, P_ind, P0_ind, G_ind, G0_ind, A_ind, A0_ind, V_ind, V0_ind, H0, N, period_last_recharge, Lambda, Min, Max, Q, CoutProd_one, out_cost_A_P_one_pipeline, period_last_recharge_pipeline, int(T), CMP

def Input_output_Time_value(chemin_instance, chemin_prodRN, chemin_prodRN_suite, chemin_prodRN_suite_suite, min_num_inst, max_num_inst):
    input0=[]
    input1=[]
    input2=[]
    input3=[]
    input4=[]
    input5=[]
    input6=[]
    input7=[]
    output_period_last_recharge=[]
    output=[]
    recalcul_T_MinQ=[]
    recalcul_T_MaxQ=[]
    recalcul_T_Delta=[]
    cout_opt=[]
    Energy_ind_mu_ind=[]
    Energy_ind_K_ind_ind=[]
    Energy_ind_R_ind=[]
    Free_ind_C=[]
    Free_ind_C_=[]
    Time_ind_R__ind=[]
    Time_ind_P__ind=[]
    Time_ind_V__ind=[]
    Time_ind_A__ind=[]
    Time_ind_G__ind=[]
    Time_ind_Delta_ind=[]
    Unit_ind_P_ind=[]
    Unit_ind_P0_ind=[]
    Unit_ind_G_ind=[]
    Unit_ind_G0_ind=[]
    Absolute_ind_A_ind=[]
    Absolute_ind_A0_ind=[]
    Absolute_ind_V_ind=[]
    Absolute_ind_V0_ind=[]
    for i in range(min_num_inst, max_num_inst):
        print(i)
        mu_ind, K_ind, R_ind, R0_ind, C_ind, C_q_ind, C__ind, I0_ind, R__ind, P__ind, V__ind, A__ind, G__ind, Iq_ind, Delta_q_ind, Delta_ind, P_ind, P0_ind, G_ind, G0_ind, A_ind, A0_ind, V_ind, V0_ind, H0, N, period_last_recharge, Lambda, Min, Max, Q, CoutProd_one, out_cost_A_P_one_pipeline, period_last_recharge_pipeline, TTTT, CMP=Indicator_of_one_instance(chemin_instance, chemin_prodRN, chemin_prodRN_suite,chemin_prodRN_suite_suite,i)
        if((Lambda<0 or Lambda>1 or period_last_recharge_pipeline<Min[Q-1] + Delta_ind or TTTT==1000000)==False):
            #if(Lambda<0 or Lambda>1 or period_last_recharge_pipeline<Min[Q-1] + Delta_ind or TTTT==1000000):
            #   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!STOP")
            input0.append([H0/mu_ind])
            input1.append([P__ind/N])
            input2.append([A__ind/N])
            input3.append([C_ind])
            input4.append([C__ind])
            input5.append([K_ind])
            input6.append([I0_ind/N])
            input7.append([1])
            output_period_last_recharge.append([period_last_recharge])
            output.append([period_last_recharge_pipeline])#Lambda
            recalcul_T_MinQ.append([Min[Q-1]])
            recalcul_T_MaxQ.append(Max[Q-1])
            recalcul_T_Delta.append([Delta_ind])
            Energy_ind_mu_ind.append([mu_ind])
            Energy_ind_K_ind_ind.append([K_ind])
            Energy_ind_R_ind.append([R_ind])
            cout_opt.append([CoutProd_one+period_last_recharge])
            Free_ind_C.append(C_ind)
            Free_ind_C_.append(C__ind)
            #R__ind, P__ind, V__ind, A__ind, G__ind,Delta_ind
            Time_ind_R__ind.append([R__ind])
            Time_ind_P__ind.append([P__ind])
            Time_ind_V__ind.append([V__ind])
            Time_ind_A__ind.append([A__ind])
            Time_ind_G__ind.append([G__ind])
            Time_ind_Delta_ind.append([Delta_ind])
            #P_ind, P0_ind, G_ind, G0_ind
            Unit_ind_P_ind.append([P_ind])
            Unit_ind_P0_ind.append([P0_ind])
            Unit_ind_G_ind.append([G_ind])
            Unit_ind_G0_ind.append([G0_ind])
            #A_ind, A0_ind, V_ind, V0_ind
            Absolute_ind_A_ind.append([A_ind])
            Absolute_ind_A0_ind.append([A0_ind])
            Absolute_ind_V_ind.append([V_ind])
            Absolute_ind_V0_ind.append([V0_ind])

    ##*************************Faire une fonction pour éviter la répétition***************************
    ##Représentation graphique des données Energy_ind
    #Nb_valtest=len(Energy_ind_K_ind_ind)
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Energy_ind_mu_ind, "y", label="Ind Mu" )
    #plt.plot(x, Energy_ind_K_ind_ind, "r", label="Ind K")
    #plt.plot(x, Energy_ind_R_ind, "b", label="Ind R" )
    #plt.plot(x, cout_opt, "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_energie.pdf')
    #plt.show()

    #    #Représentation graphique des 90 données Energy_ind
    #Nb_valtest=len(Energy_ind_K_ind_ind[0:90])
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Energy_ind_mu_ind[0:90], "y", label="Ind Mu" )
    #plt.plot(x, Energy_ind_K_ind_ind[0:90], "r", label="Ind K")
    #plt.plot(x, Energy_ind_R_ind[0:90], "b", label="Ind R" )
    #plt.plot(x, cout_opt[0:90], "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_energie_90.pdf')
    #plt.show()

    #    #Représentation graphique des données Free_ind
    #Nb_valtest=len(Free_ind_C)
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Free_ind_C, "y", label="Ind C" )
    #plt.plot(x, Free_ind_C_, "r", label="Ind C*")
    #plt.plot(x, cout_opt, "g", label="Opt" )#[0:90]
    #plt.legend()
    ##plt.savefig('Indicateurs_free.pdf')
    #plt.show()

    #    #Représentation graphique des 90 données Free_ind
    #Nb_valtest=len(Free_ind_C[0:90])
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Free_ind_C[0:90], "y", label="Ind C" )
    #plt.plot(x, Free_ind_C_[0:90], "r", label="Ind C*")
    #plt.plot(x, cout_opt[0:90], "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_free_90.pdf')
    #plt.show()


    ##Représentation graphique des données Time_ind
    ##R__ind, P__ind, V__ind, A__ind, Delta_ind
    #Nb_valtest=len(Time_ind_R__ind)
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Time_ind_R__ind, "y", label="Ind R*" )
    #plt.plot(x, Time_ind_P__ind, "r", label="Ind P*")
    #plt.plot(x, Time_ind_V__ind, "b", label="Ind V*" )
    #plt.plot(x, Time_ind_A__ind, "p", label="Ind A*")
    ##plt.plot(x, Time_ind_G__ind, "o", label="Ind G*" )
    #plt.plot(x, Time_ind_Delta_ind, "o", label="Ind Delta")
    #plt.plot(x, cout_opt, "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_time.pdf')
    #plt.show()

    ##G__ind,
    #plt.plot(x, Time_ind_G__ind, "o", label="Ind G*" )
    #plt.legend()
    #plt.savefig('Indicateurs_time_G.pdf')
    #plt.show()



    #    #Représentation graphique des 90 données Time_ind
    #Nb_valtest=len(Time_ind_R__ind[0:90])
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Time_ind_R__ind[0:90], "y", label="Ind R*" )
    #plt.plot(x, Time_ind_P__ind[0:90], "r", label="Ind P*")
    #plt.plot(x, Time_ind_V__ind[0:90], "b", label="Ind V*" )
    #plt.plot(x, Time_ind_A__ind[0:90], "p", label="Ind A*")
    #plt.plot(x, Time_ind_G__ind[0:90], "o", label="Ind G*" )
    #plt.plot(x, Time_ind_Delta_ind[0:90], "b", label="Ind Delta")
    #plt.plot(x, cout_opt[0:90], "g", label="Opt" )#[0:90]
    #plt.legend()
    ##plt.savefig('Indicateurs_time_90.pdf')
    #plt.show()

    ##G__ind,
    #plt.plot(x, Time_ind_G__ind[0:90], "o", label="Ind G*" )
    #plt.legend()
    ##plt.savefig('Indicateurs_time_G_90.pdf')
    #plt.show()


    ##Représentation graphique des données Unit_ind
    ##P_ind, P0_ind, G_ind, G0_ind
    #Nb_valtest=len(Unit_ind_P_ind)
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Unit_ind_P_ind, "y", label="Ind P" )
    #plt.plot(x, Unit_ind_P0_ind, "r", label="Ind P0")
    #plt.plot(x, Unit_ind_G_ind, "b", label="Ind G" )
    #plt.plot(x, Unit_ind_G0_ind, "p", label="Ind G0")
    #plt.plot(x, cout_opt, "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_unit.pdf')
    #plt.show()

    #    #Représentation graphique des 90 données Unit_ind
    #Nb_valtest=len(Unit_ind_P_ind[0:90])
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Unit_ind_P_ind[0:90], "y", label="Ind P" )
    #plt.plot(x, Unit_ind_P0_ind[0:90], "r", label="Ind P0")
    #plt.plot(x, Unit_ind_G_ind[0:90], "b", label="Ind G" )
    #plt.plot(x, Unit_ind_G0_ind[0:90], "p", label="Ind G0")
    #plt.plot(x, cout_opt[0:90], "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_unit_90.pdf')
    #plt.show()

    ##Représentation graphique des données Absolute_ind
    # #A_ind, A0_ind, V_ind, V0_ind
    #Nb_valtest=len(Absolute_ind_A_ind)
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Absolute_ind_A_ind, "y", label="Ind A" )
    #plt.plot(x, Absolute_ind_A0_ind, "r", label="Ind A0")
    #plt.plot(x, Absolute_ind_V_ind, "b", label="Ind V" )
    #plt.plot(x, Absolute_ind_V0_ind, "p", label="Ind V0")
    #plt.plot(x, cout_opt, "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_absolute.pdf')
    #plt.show()

    #    #Représentation graphique des 90 données Absolute_ind
    #Nb_valtest=len(Absolute_ind_A_ind[0:90])
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, Absolute_ind_A_ind[0:90], "y", label="Ind A" )
    #plt.plot(x, Absolute_ind_A0_ind[0:90], "r", label="Ind A0")
    #plt.plot(x, Absolute_ind_V_ind[0:90], "b", label="Ind V" )
    #plt.plot(x, Absolute_ind_V0_ind[0:90], "p", label="Ind V0")
    #plt.plot(x, cout_opt[0:90], "g", label="Opt" )#[0:90]
    #plt.legend()
    #plt.savefig('Indicateurs_absolute_90.pdf')
    #plt.show()

    #*************************Faire une fonction pour éviter la répétition***************************
    cheminComplet_test = "cinq_instances_pr_test.txt"
    concat_input_TV=str(output[70-70])+ " "+ str(output[84-70]) + " " + str(output[89-70])+ " "+ str(output[94-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("Lambda pipeline des instances : " + str(70)+str(84)+str(89)+str(94) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="H0/mu="+str(input0[70-70]) +" P0/N="+str(input1[70-70])+ " A0/N="+str(input2[70-70])+" C="+str(input3[70-70])+ " C*=" + str(input4[70-70])+" K="+ str(input5[70-70])+ " I0/N=" +str(input6[70-70])+ "Lambda=" +str(input7[70-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(70) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="H0/mu="+str(input0[84-70]) +" P0/N="+str(input1[84-70])+ " A0/N="+str(input2[84-70])+" C="+str(input3[84-70])+ " C*=" + str(input4[84-70])+" K="+ str(input5[84-70])+ " I0/N=" +str(input6[84-70])+ "Lambda=" +str(input7[84-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(84) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="H0/mu="+str(input0[89-70]) +" P0/N="+str(input1[89-70])+ " A0/N="+str(input2[89-70])+" C="+str(input3[89-70])+ " C*=" + str(input4[89-70])+" K="+ str(input5[89-70])+ " I0/N=" +str(input6[89-70])+ "Lambda=" +str(input7[89-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(89) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="H0/mu="+str(input0[94-70]) +" P0/N="+str(input1[94-70])+ " A0/N="+str(input2[94-70])+" C="+str(input3[94-70])+ " C*=" + str(input4[94-70])+" K="+ str(input5[94-70])+ " I0/N=" +str(input6[94-70])+ "Lambda=" +str(input7[94-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(94) + "\n")
        fichier_RSS.write(concat_input_TV + "\n") 
    return input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge, output, recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta

def creationPoidsSetup(N):
	biais = np.zeros(N)
	poids = np.eye(N)#matrice identite

	#-1 sur les cases (i,i+1)
	for i in range(0,N-1):
		poids[i][i+1] = -1

	return(poids, biais)

def affichePoids(model, name, use_biais):

    cheminComplet = "Parameters_alternative_learning.txt"
    with open(cheminComplet, "a") as fichier_RS:
        print ("affichage des poids de la couche : ", name)
        fichier_RS.write("affichage des poids de la couche : "+ name)

        print ("\n")
        fichier_RS.write("\n")

        #afficher les poids qui arrive sur une couche
        WS=model.get_layer(name).get_weights()

        print("weights = ", WS[0])
        fichier_RS.write("weights = \n")
        print(type(WS[0]))
       # fichier_RS.write(WS[0])
        np.savetxt(fichier_RS, WS[0])

        if(use_biais == True):
            print("biais = ", WS[1])
            fichier_RS.write("biais = \n")
           # fichier_RS.write(WS[1])
            np.savetxt(fichier_RS, WS[1])

        print ("\n")
        fichier_RS.write("\n")
        return WS[0]

def Time_value_model(input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge,recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta, output):
    #---------------------------------------------------------------------------------------------------
    #ETAPE 1. on va definir chaque neurones de la couche input separement, on les stocke dans un tableau
    MinQDelta = keras.Input(shape=(2,), name="MinQDelta")
    MaxQ = keras.Input(shape=(1,), name="MaxQ")

    coucheInput = []
    for i in range(8):
        input_tmp = keras.Input(shape=(1,), name="input"+str(i)) 
        coucheInput.append(input_tmp)

    couche1_A=[]

    for i in range(7):
        tmp = keras.layers.Dense(1, activation='linear', name="TV_lambda"+str(i), kernel_constraint=keras.constraints.NonNeg(), use_bias=False)(coucheInput[i])
        couche1_A.append(tmp)
    tmp = keras.layers.Dense(1, activation='linear', name="TV_lambda"+str(7), use_bias=False)(coucheInput[7])#ici ce biais est non signé
    couche1_A.append(tmp)
    concat = keras.layers.concatenate(couche1_A)

    #(poids_setup, biais_setup) = creationPoidsSetup(7)
    #le dernier parmètre est le billet
    poids_setup=np.array([[1],[1],[1],[1],[1],[-1],[-1],[-1]])
    biais_setup=np.array([0])
    out_val1 = keras.layers.Dense(1, activation='linear', name="out_val1", weights=[poids_setup, biais_setup], trainable=False)(concat)
    #paramètre : gamma
    #sortie : lambda
    out_val = keras.layers.Dense(1, activation='sigmoid', name="TV_gamma", kernel_constraint=keras.constraints.NonNeg(), use_bias=False)(out_val1)

    poids_setup5=np.array([[1]])
    biais_setup5=np.array([0])
    #sortie : lambda
    couche_lambda_B_1 = keras.layers.Dense(1, activation='linear', name="double_lambda_1", weights=[poids_setup5, biais_setup5], trainable=False)(out_val)
    #sortie : alpha
    couche_lambda_B_2 = keras.layers.Dense(1, activation='linear', name="double_lambda_2", weights=[poids_setup5, biais_setup5], trainable=False)(out_val)
    couche_lambda_B = keras.layers.concatenate([couche_lambda_B_1, couche_lambda_B_2])
    Multp_2 = keras.layers.Multiply()([MinQDelta, couche_lambda_B])
    
    concat3 = keras.layers.concatenate([Multp_2, MaxQ])

    poids_setup6=np.array([[1],[-1],[1]])
    biais_setup6=np.array([0])
    T_val = keras.layers.Dense(1, activation='linear', name="couche_T", weights=[poids_setup6, biais_setup6], trainable=False)(concat3)


    #---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
    in_all = [MinQDelta,MaxQ]
    in_all += coucheInput
    model = keras.Model(inputs=in_all, outputs=T_val)
    #model = keras.Model(inputs=coucheInput, outputs=T_val)
    
	#résumé du modèle
    model.summary()
    
	#dessin du modele
    keras.utils.plot_model(model, "reseauALternativeLearning.png", show_shapes=True, dpi=192)
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    #---------------------------------------------------------------------------------------------------
	# ETAPE 3.  entrainement du modele

    my_input_pred=[]
    y_train2 = np.asarray(output[90:])#
    
    for j in range(90,y_train2.shape[0]+90):##données d'entrainements sans les données de test 90,y_train2.shape[0]+90.............0,y_train2.shape[0]
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input = np.asarray(my_input_pred)
        
    #input=[]
    #for i in range(8):
    #    input.append(my_input[:,i])
    MinQDelta_MaxQ2=[]
    for i in range(90,y_train2.shape[0]+90):
        MinQDelta_MaxQ2.append([recalcul_T_MinQ[i][0]+recalcul_T_Delta[i][0],recalcul_T_MaxQ[i]])
    input={"MinQDelta": np.asarray(MinQDelta_MaxQ2), "MaxQ": np.asarray(recalcul_T_MaxQ[90:]), "input0": my_input[:,0],"input1": my_input[:,1],"input2": my_input[:,2],"input3": my_input[:,3],"input4": my_input[:,4],"input5": my_input[:,5],"input6": my_input[:,6],"input7": my_input[:,7]}
	#entrainement
    history = model.fit(input, y_train2, validation_split=0.33, epochs=200, batch_size=32, verbose=0)
    #print(history.history)
    

    # on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
    plt.plot(history.history["loss"])#history.history["loss"][-1]: dernier élément de cette liste
    plt.plot(history.history["val_loss"])
    plt.title("erreur en fonction des époques")
    plt.ylabel('loss (carré moyen des erreurs)')
    plt.xlabel('nombre d\'époques')
    plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
    plt.savefig('loss_reseauALternativeLearning.pdf')
    plt.show()
    
    #Vérifie si le loss est bien calculé
    #som=0
    #for i in range(0, len(y_train2)):
    #    som=som+((y_train2[i] - y[i][0])**2)
    #loss_recalculate=som/len(y_train2)
    #real_loss=history.history["loss"][-1]
    #if(real_loss==loss_recalculate):
    #    print("***********************************************\n")
    #    print("Le loss est bien calculé")

    model.save('my_model_learn_time.h5')  # creates a HDF5 file 'my_model.h5'
    #del model  # deletes the existing model

def Time_value_predict(input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge,recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta, output):

    model = load_model('my_model_learn_time.h5')
    
    my_input_pred=[]
    y_train2 = np.asarray(output[90:])#
    
    for j in range(90,y_train2.shape[0]+90):##données d'entrainements sans les données de test 90,y_train2.shape[0]+90.............0,y_train2.shape[0]
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input = np.asarray(my_input_pred)
        
    #input=[]
    #for i in range(8):
    #    input.append(my_input[:,i])
    MinQDelta_MaxQ2=[]
    for i in range(90,y_train2.shape[0]+90):
        MinQDelta_MaxQ2.append([recalcul_T_MinQ[i][0]+recalcul_T_Delta[i][0],recalcul_T_MaxQ[i]])
    input={"MinQDelta": np.asarray(MinQDelta_MaxQ2), "MaxQ": np.asarray(recalcul_T_MaxQ[90:]), "input0": my_input[:,0],"input1": my_input[:,1],"input2": my_input[:,2],"input3": my_input[:,3],"input4": my_input[:,4],"input5": my_input[:,5],"input6": my_input[:,6],"input7": my_input[:,7]}
    
    #Représentation graphique des données d'apprentissage
    #représentation graphique en une unique courbe
    Nb_valtest=y_train2.shape[0]
    y = model.predict(input)
    k_itt=90
    err=0
    for i in range(Nb_valtest):
        if(output[i]!=0):
            err = err + (abs(output[k_itt] - y[i]) * 100) / output[k_itt] # on calcule l'erreur realisee par le reseau de neurones
        k_itt=k_itt+1
    print("erreur train = " + str(err / Nb_valtest) + "%")
    #for i in range(90,Nb_valtest):#ceci était valable lorsqu'on calaculait lambda
    #    y[i][0]=y[i][0]*(recalcul_T_MinQ[i][0] + recalcul_T_Delta[i][0]) +(1-y[i][0])*recalcul_T_MaxQ[i]
    y_train_temp=output[90:]
    y_train_temp_tri=y_train_temp
    y_tri=y
    y_train_temp_tri, y_tri =zip(*sorted(zip(y_train_temp_tri, y_tri)))
    
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur

    y_train_temp_tri_by_100=[]#y_train_temp_tri_by_100 est la liste obtenues après transformation de la liste y_train_temp_tri en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        y_train_temp_tri_by_100.append(np.mean(y_train_temp_tri[100*k:100*k+100]))
    y_train_temp_tri_by_100.append(np.mean(y_train_temp_tri[5900:5909]))

    y_tri_by_100=[]#y_tri_by_100 est la liste obtenues après transformation de la liste y_tri en la regroupant par paquet de 100 
    #x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        y_tri_by_100.append(np.mean(y_tri[100*k:100*k+100]))
    y_tri_by_100.append(np.mean(y_tri[5900:5909]))

    plt.plot(x, y_train_temp_tri_by_100, "g+",label="Numéro de période de la dernière recharge de la tournée optimale")
    plt.plot(x, y_tri_by_100, "ro", label="Numéro de période de la dernière recharge de la tournée prédite")
    plt.ylabel('Numéro de période de la dernière recharge')
    plt.xlabel('numéro de l\'instance')
    plt.legend()
    plt.savefig('prediction_reseauALternativeLearning_train_data.pdf')
    plt.show()
    

    #Représentation graphique des données de test
    my_input_pred=[]
    y_train2 = np.asarray(output[0:90])#
    
    for j in range(0,90):##données d'entrainements sans les données de test 90,y_train2.shape[0]+90.............0,y_train2.shape[0]
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input_test = np.asarray(my_input_pred)
    #input_test=[]
    #for i in range(8):
    #    input_test.append(my_input_test[:,i])
    MinQDelta_MaxQ2=[]
    #for j in range(0,90):
    #    MinQDelta_MaxQ2.append(recalcul_T_MinQ[i]+recalcul_T_Delta[i])
    for i in range(0,90):
        MinQDelta_MaxQ2.append([recalcul_T_MinQ[i][0]+recalcul_T_Delta[i][0],recalcul_T_MaxQ[i]])
    input_test={"MinQDelta": np.asarray(MinQDelta_MaxQ2), "MaxQ": np.asarray(recalcul_T_MaxQ[0:90]), "input0": my_input[0:90,0],"input1": my_input[0:90,1],"input2": my_input[0:90,2],"input3": my_input[0:90,3],"input4": my_input[0:90,4],"input5": my_input[0:90,5],"input6": my_input[0:90,6],"input7": my_input[0:90,7]}
    y = model.predict(input_test) #prédiction des valeurs lambda
    #for i in range(0,90):
     #   y[i][0]=y[i][0]*(recalcul_T_MinQ[i][0] + recalcul_T_Delta[i][0]) +(1-y[i][0])*recalcul_T_MaxQ[i]
    y_train_temp=output[0:90]
    y_train_temp_tri=y_train_temp
    y_tri=y
    y_train_temp_tri, y_tri =zip(*sorted(zip(y_train_temp_tri, y_tri)))
    Nb_valtest=90
    err=0
    for i in range(Nb_valtest):
        err = err + (abs(y[i] - y_train_temp[i]) * 100) / y[i] # on calcule l'erreur realisee par le reseau de neurones
    print("erreur test= " + str(err / Nb_valtest) + "%")

    
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    plt.plot(x, y_train_temp_tri, "g+", label="Numéro de période de la dernière recharge de la tournée optimale" )
    plt.plot(x, y_tri, "ro", label="Numéro de période de la dernière recharge de la tournée prédite")
    plt.ylabel('Numéro de période de la dernière recharge')
    plt.xlabel('numéro de l\'instance')
    plt.legend()
    plt.savefig('prediction_reseauALternativeLearning_test_data.pdf')
    plt.show()

    #affichage des poids des couches
    lambd0=affichePoids(model, "TV_lambda0", False)
    lambd1=affichePoids(model, "TV_lambda1", False)
    lambd2=affichePoids(model, "TV_lambda2", False)
    lambd3=affichePoids(model, "TV_lambda3", False)
    lambd4=affichePoids(model, "TV_lambda4", False)
    lambd5=affichePoids(model, "TV_lambda5", False)
    lambd6=affichePoids(model, "TV_lambda6", False)
    lambd7=affichePoids(model, "TV_lambda7", False)
    gamm=affichePoids(model, "TV_gamma", False)

    
    #affichePoids(model,"out_val1", True)
    #Si on connait mettre les bornes sur les paramètres on utilise ce réseau:

    #------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------lambda
    #courbe du lambda opt, du lambda prédit et du lambda recalculé
   
    y_train_pr_recalcul = np.asarray(output)#

    Nb_valtest=y_train_pr_recalcul.shape[0]
    my_input_pred=[]
    for j in range(0,Nb_valtest):
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input = np.asarray(my_input_pred)
        
    #input_pr_recalcul=[]
    #for i in range(8):
    #    input_pr_recalcul.append(my_input[:,i])
    MinQDelta_MaxQ2=[]
    for i in range(0,Nb_valtest):
        MinQDelta_MaxQ2.append([recalcul_T_MinQ[i][0]+recalcul_T_Delta[i][0],recalcul_T_MaxQ[i]])
    input_pr_recalcul={"MinQDelta": np.asarray(MinQDelta_MaxQ2), "MaxQ": np.asarray(recalcul_T_MaxQ[:]), "input0": my_input[:,0],"input1": my_input[:,1],"input2": my_input[:,2],"input3": my_input[:,3],"input4": my_input[:,4],"input5": my_input[:,5],"input6": my_input[:,6],"input7": my_input[:,7]}
    couche_lambda = keras.Model(inputs=model.input, outputs=model.get_layer("TV_gamma").output)
    y_lambda = couche_lambda(input_pr_recalcul)
    #y = model.predict(input_pr_recalcul)

    #y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    y_recalcule=[]
    for i in range(0,Nb_valtest):
        X=lambd0*input0[i]+lambd1*input1[i]+lambd2*input2[i]+lambd3*input3[i]+lambd4*input4[i]-lambd5*input5[i]-lambd6*input6[i]-lambd7*input7[i]
        y_recalcule.append([exp(gamm*X)/(exp(gamm*X)+1)])
    
    y_train_temp_out=output
    y_2=y_lambda
    #y_1=y_lambda
   
    #y_train_temp_out, y_1 =zip(*sorted(zip(y_train_temp_out, y_1)))
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, y_train_temp_out, "g", label="lambda_Opt" )
    #plt.plot(x, y_1, "r", label="lambda_Predict")
    #plt.legend()
    #plt.savefig('lambda_prediction_reseauALternativeLearning_train_and_test_data.png')
    #plt.show()
    y_recalcule_tri=y_recalcule
    y_2_tri=y_2
    y_recalcule_tri, y_2_tri =zip(*sorted(zip(y_recalcule_tri, y_2_tri)))
    plt.plot(x, y_2_tri, "r", label="lambda_Predict")
    plt.plot(x, y_recalcule_tri, "y", label="lambda_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('lambda_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()

    #---------------------------------------------------------------------T
     #courbe du T opt, du T prédit et du T recalculé

    T_recalcule=[]
    for i in range(0,Nb_valtest):
        T_recalcule.append([(y_lambda[i][0]*(recalcul_T_MinQ[i][0]+recalcul_T_Delta[i][0]))+((1-y_lambda[i][0])*recalcul_T_MaxQ[i])])

    #for i in range(0,Nb_valtest):
    #   y[i][0]=y[i][0]*(recalcul_T_MinQ[i][0] + recalcul_T_Delta[i][0]) +(1-y[i][0])*recalcul_T_MaxQ[i]
    y_train_temp=output
    #y_1=y
    y_2=model.predict(input_pr_recalcul)
    #y_train_temp, y_1 =zip(*sorted(zip(y_train_temp, y_1)))
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, y_train_temp, "g", label="T_Opt" )
    #plt.plot(x, y_1, "r", label="T_Predict")
    #plt.legend()
    #plt.savefig('T_prediction_reseauALternativeLearning_train_and_test_data.png')
    #plt.show()
    T_recalcule_tri=T_recalcule
    y_2_tri=y_2
    T_recalcule_tri, y_2_tri =zip(*sorted(zip(T_recalcule_tri, y_2_tri)))
    plt.plot(x, y_2_tri, "r", label="T_Predict")
    plt.plot(x, T_recalcule_tri, "y", label="T_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('T2_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()


def Input_output_Cost_value(chemin_instance, chemin_prodRN, chemin_prodRN_suite, chemin_prodRN_suite_suite, min_num_inst, max_num_inst):
    in_PR_A_RP0_A_RG_A = []
    in_A_PR_A_N = []
    in_A_moins_A0_A_plus_A0 = []
    in_A_plus_A0 = []
    in_P_moins_P0_P_plus_P0 = []
    in_P_plus_P0 = []
    in_K = []
    in_mu = []
    out_Cost_A_P = []
    input0 = []
    input1 = []
    input2 = []
    input3 = []
    input4 = []
    input5 = []
    CMP_list = []
    Nb_inst=0;
    for i in range(min_num_inst, max_num_inst):
        print(i)
        mu_ind, K_ind, R_ind, R0_ind, C_ind, C_q_ind, C__ind, I0_ind, R__ind, P__ind, V__ind, A__ind, G__ind, Iq_ind, Delta_q_ind, Delta_ind, P_ind, P0_ind, G_ind, G0_ind, A_ind, A0_ind, V_ind, V0_ind, H0, N, period_last_recharge, Lambda, Min, Max, Q, CoutProd_one, out_cost_A_P_one_pipeline, period_last_recharge_pipeline,TTTT, CMP=Indicator_of_one_instance(chemin_instance, chemin_prodRN, chemin_prodRN_suite,chemin_prodRN_suite_suite,i)
        Nb_inst=Nb_inst+1;
       #,out_cost_A_P_one_pipeline, period_last_recharge_pipeline
        if((Lambda<0 or Lambda>1 or period_last_recharge_pipeline<Min[Q-1] + Delta_ind or TTTT==1000000)==False):
            if(Lambda<0 or Lambda>1 or period_last_recharge_pipeline<Min[Q-1] + Delta_ind or TTTT==1000000):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!STOP")
            in_PR_A_RP0_A_RG_A.append([(P_ind*R_ind)/A_ind,(R_ind*P0_ind)/A_ind,(R_ind*G_ind)/A_ind])
            in_A_PR_A_N.append([A_ind/(P_ind*R_ind),A__ind/N,-1])
            in_A_moins_A0_A_plus_A0.append([A_ind-A0_ind,A_ind+A0_ind])
            in_A_plus_A0.append([A_ind+A0_ind])#
            in_P_moins_P0_P_plus_P0.append([P_ind-P0_ind,P_ind+P0_ind])
            in_P_plus_P0.append([P_ind+P0_ind])
            in_K.append([K_ind])
            in_mu.append([mu_ind])
            out_Cost_A_P.append([out_cost_A_P_one_pipeline])#out_cost_A_P_one_pipeline=K_ind*cost_A+mu_ind*cost_P 
            input0.append([P__ind/N])
            input1.append([A_ind/(P_ind*R_ind)])
            input2.append([G_ind/P_ind])
            input3.append([V0_ind/V_ind])
            input4.append([C_ind])
            input5.append([1])
            CMP_list.append([CMP])
    
    cheminComplet_test = "cinq_instances_pr_test.txt"
    concat_input_TV=str(out_Cost_A_P[70-70])+ " "+ str(out_Cost_A_P[84-70]) + " " + str(out_Cost_A_P[89-70])+ " "+ str(out_Cost_A_P[94-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("COST pipeline des instances : " + str(70)+str(84)+str(89)+str(94) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="(P_ind*R_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[70-70][0]) +" (R_ind*P0_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[70-70][1])+ " (R_ind*G_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[70-70][2])+" A_ind/(P_ind*R_ind)="+str(in_A_PR_A_N[70-70][0])+ " A__ind/N =" + str(in_A_PR_A_N[70-70][1])+" A_ind-A0_ind="+ str(in_A_moins_A0_A_plus_A0[70-70][0])+ " A_ind+A0_ind=" +str(in_A_moins_A0_A_plus_A0[70-70][1])+ "P_ind-P0_ind=" +str(in_P_moins_P0_P_plus_P0[70-70][0]) + "P_ind+P0_ind=" + str(in_P_moins_P0_P_plus_P0[70-70][1])+ " K_ind=" + str(in_K[70-70])+ " mu_ind=" + str(in_mu[70-70])+ " P0_ind/N=" + str(input0[70-70]) + " A_ind/(P_ind*R_ind)=" + str(input1[70-70])+ " G_ind/P_ind=" + str(input2[70-70])+ " V0_ind/V_ind=" + str(input3[70-70])+ " C_ind=" + str(input4[70-70])+ "alpha_0=" + str(input5[70-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(70) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="(P_ind*R_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[84-70][0]) +" (R_ind*P0_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[84-70][1])+ " (R_ind*G_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[84-70][2])+" A_ind/(P_ind*R_ind)="+str(in_A_PR_A_N[84-70][0])+ " A__ind/N =" + str(in_A_PR_A_N[84-70][1])+" A_ind-A0_ind="+ str(in_A_moins_A0_A_plus_A0[84-70][0])+ " A_ind+A0_ind=" +str(in_A_moins_A0_A_plus_A0[84-70][1])+ "P_ind-P0_ind=" +str(in_P_moins_P0_P_plus_P0[84-70][0]) + "P_ind+P0_ind=" +str(in_P_moins_P0_P_plus_P0[84-70][1])+ " K_ind=" + str(in_K[84-70])+ " mu_ind=" + str(in_mu[84-70]) + " P0_ind/N=" + str(input0[84-70]) + " A_ind/(P_ind*R_ind)=" + str(input1[84-70])+ " G_ind/P_ind=" + str(input2[84-70])+ " V0_ind/V_ind=" + str(input3[84-70])+ " C_ind=" + str(input4[84-70])+ "alpha_0=" + str(input5[84-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(84) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="(P_ind*R_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[89-70][0]) +" (R_ind*P0_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[89-70][1])+ " (R_ind*G_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[89-70][2])+" A_ind/(P_ind*R_ind)="+str(in_A_PR_A_N[89-70][0])+ " A__ind/N =" + str(in_A_PR_A_N[89-70][1])+" A_ind-A0_ind="+ str(in_A_moins_A0_A_plus_A0[89-70][0])+ " A_ind+A0_ind=" +str(in_A_moins_A0_A_plus_A0[89-70][1])+ "P_ind-P0_ind=" +str(in_P_moins_P0_P_plus_P0[89-70][0]) + "P_ind+P0_ind=" +str(in_P_moins_P0_P_plus_P0[89-70][1])+ " K_ind=" + str(in_K[89-70])+ " mu_ind=" + str(in_mu[89-70])+ " P0_ind/N=" + str(input0[89-70])+ " A_ind/(P_ind*R_ind)=" + str(input1[89-70]) + " G_ind/P_ind=" + str(input2[89-70])+ " V0_ind/V_ind=" + str(input3[89-70]) + " C_ind=" + str(input4[89-70])+ "alpha_0=" + str(input5[89-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(89) + "\n")
        fichier_RSS.write(concat_input_TV + "\n")
    concat_input_TV="(P_ind*R_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[94-70][0]) +" (R_ind*P0_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[94-70][1])+ " (R_ind*G_ind)/A_ind="+str(in_PR_A_RP0_A_RG_A[94-70][2])+" A_ind/(P_ind*R_ind)="+str(in_A_PR_A_N[94-70][0])+ " A__ind/N =" + str(in_A_PR_A_N[94-70][1])+" A_ind-A0_ind="+ str(in_A_moins_A0_A_plus_A0[94-70][0])+ " A_ind+A0_ind=" +str(in_A_moins_A0_A_plus_A0[94-70][1])+ "P_ind-P0_ind=" +str(in_P_moins_P0_P_plus_P0[94-70][0]) + "P_ind+P0_ind=" +str(in_P_moins_P0_P_plus_P0[94-70][1]) + " K_ind=" + str(in_K[94-70])+ " mu_ind=" + str(in_mu[94-70])+ " P0_ind/N=" + str(input0[94-70])+ " A_ind/(P_ind*R_ind)=" + str(input1[94-70])+ " G_ind/P_ind=" + str(input2[94-70])+ " V0_ind/V_ind=" + str(input3[94-70])+ " C_ind=" + str(input4[94-70])+ "alpha_0=" + str(input5[94-70])
    with open(cheminComplet_test, "a") as fichier_RSS:
        fichier_RSS.write("instance " + str(94) + "\n")
        fichier_RSS.write(concat_input_TV + "\n") 
    return in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list

def Cost_value_model(in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list ):
    
    #debug initializer = tf.keras.initializers.RandomUniform(minval=1, maxval=5)
    
     #---------------------------------Partie du code qui évalue COST_A--------------------------------------------------------------

    my_input1 = keras.Input(shape=(3,), name="in1")#PR_A_RP0_A_RG_A
    my_input2 = keras.Input(shape=(3,), name="in2")#in_A_PR_A_N
    my_input3 = keras.Input(shape=(2,), name="in3")#A_moins_A0_A_plus_A0
    my_input4 = keras.Input(shape=(1,), name="in4")#A_plus_A0
    my_input5 = keras.Input(shape=(2,), name="in5")#P_moins_P0_P_plus_P0
    my_input6 = keras.Input(shape=(1,), name="in6")#P_plus_P0
    my_input7 = keras.Input(shape=(1,), name="K")#corresponds à K et mu des indicateurs
    my_input8 = keras.Input(shape=(1,), name="mu")
    my_input9 = keras.Input(shape=(1,), name="input55")
    #paramètres : tau_1, tau_2, tau_3
    #sortie : tau
    couche1_A = keras.layers.Dense(1, activation='linear', name="tau", use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomUniform(minval=3, maxval=4)
                                   )(my_input1)
    poids_setup1=np.array([[1],[1]])
    biais_setup1=np.array([0])
    #sortie : tau+1
    concat0 = keras.layers.concatenate([couche1_A, my_input9])
    couche2_A = keras.layers.Dense(1, activation='linear', name="couche2_A", weights=[poids_setup1, biais_setup1], trainable=False)(concat0)
    
    #paramètres : beta_1, beta_2, beta_0
    #sortie : Y
    couche1_B = keras.layers.Dense(1, activation='linear', name="beta", kernel_constraint=keras.constraints.NonNeg(), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=1/2, maxval=3)
                                   )(my_input2)
    
    #paramètre : omega
    #sortie : beta
    couche2_B =  keras.layers.Dense(1, activation='sigmoid', name="omega", use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=1)
                                    )(couche1_B)
    
    poids_setup2=np.array([[1]])
    biais_setup2=np.array([0])
    #paramètres : 1
    #sortie : beta
    couche3_B_1 = keras.layers.Dense(1, activation='linear', name="couche3_B_1", weights=[poids_setup2, biais_setup2], trainable=False)(couche2_B)
    #paramètres : 1
    #sortie : beta
    couche3_B_2 = keras.layers.Dense(1, activation='linear', name="couche3_B_2", weights=[poids_setup2, biais_setup2], trainable=False)(couche2_B)
    couche3_B = keras.layers.concatenate([couche3_B_1, couche3_B_2])

    Mulp_1 = keras.layers.Multiply()([my_input3, couche3_B])
    
    concat1 = keras.layers.concatenate([Mulp_1, my_input4])

    poids_setup3=np.array([[1],[-1],[1]])
    biais_setup3=np.array([0])
    couche4_B = keras.layers.Dense(1, activation='linear', name="couche4_B", weights=[poids_setup3, biais_setup3], trainable=False)(concat1)
    
    COST_AA = keras.layers.Multiply()([couche2_A, couche4_B])

    poids_setup2=np.array([[1]])
    biais_setup2=np.array([0])
    #paramètres : 1
    #sortie : COST_A
    COST_A = keras.layers.Dense(1, activation='linear', name="COST_A_n", weights=[poids_setup2, biais_setup2], trainable=False)(COST_AA)
    
    #---------------------------------Partie du code qui évalue COST_P--------------------------------------------------------------

    coucheInput_cost_P = []
    #input0 = P0/N
    #input1 = A/P*R
    #input2 = G/P
    #input3 = V0/V
    #input4 = C
    #input5 = 1 
    for i in range(6):
        input_tmp = keras.Input(shape=(1,), name="input"+str(i)) 
        coucheInput_cost_P.append(input_tmp)

    couche1_A_cost_P=[]
    #coucheInput_cost_P[0] ------ l'input de cette couche est le vecteur de P0/N correspondant à alpha_0 ---- alpha1 de l'article
    #coucheInput_cost_P[1] ------ l'input de cette couche est le vecteur de A/P*R correspondant à alpha_1 ---- alpha2 de l'article
    #coucheInput_cost_P[2] ------ l'input de cette couche est le vecteur de G/P correspondant à alpha_2 ---- alpha3 de l'article
    #coucheInput_cost_P[3] ------ l'input de cette couche est le vecteur de V0/V correspondant à alpha_3 ---- alpha4 de l'article
    #coucheInput_cost_P[4] ------ l'input de cette couche est le  de C correspondant à alpha_4 ---- alpha5 de l'article
    #coucheInput_cost_P[5] ------ l'input de cette couche est le vecteur de 1 correspondant à alpha_5 ---- alpha0 de l'article
    for i in range(5):
        tmp = keras.layers.Dense(1, activation='linear',name="alpha"+str(i), use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomUniform(minval=1/2, maxval=1)
                                 )(coucheInput_cost_P[i])
        couche1_A_cost_P.append(tmp)
    tmp = keras.layers.Dense(1, activation='linear',name="alpha"+str(5), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0)
                             )(coucheInput_cost_P[5])
    couche1_A_cost_P.append(tmp)
    concat2 = keras.layers.concatenate(couche1_A_cost_P)

    #le dernier parmètre est le billet
    poids_setup4=np.array([[1],[-1],[-1],[-1],[-1],[-1]])
    biais_setup4=np.array([0])
    #sortie : Z
    Z = keras.layers.Dense(1, activation='linear', name="Z", weights=[poids_setup4, biais_setup4], trainable=False)(concat2)
    #paramètres: sigma
    #sortie : alpha
    alpha = keras.layers.Dense(1, activation='sigmoid', name="sigma", kernel_constraint=keras.constraints.NonNeg(), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=1)
                               )(Z)

    poids_setup5=np.array([[1]])
    biais_setup5=np.array([0])
    #sortie : alpha
    couche5_B_1 = keras.layers.Dense(1, activation='linear', name="double_alpha_1", weights=[poids_setup5, biais_setup5], trainable=False)(alpha)
    #sortie : alpha
    couche5_B_2 = keras.layers.Dense(1, activation='linear', name="double_alpha_2", weights=[poids_setup5, biais_setup5], trainable=False)(alpha)
    couche5_B = keras.layers.concatenate([couche5_B_1, couche5_B_2])

    Mulp_2 = keras.layers.Multiply()([my_input5, couche5_B])
    
    concat3 = keras.layers.concatenate([Mulp_2, my_input6])

    poids_setup6=np.array([[1],[-1],[1]])
    biais_setup6=np.array([0])
    COST_P = keras.layers.Dense(1, activation='linear', name="couche6_B", weights=[poids_setup6, biais_setup6], trainable=False)(concat3)
    
    #---------------------------------Partie du code qui évalue COST--------------------------------------------------------------

    Mulp_3 = keras.layers.Multiply()([my_input7, COST_A])
    Mulp_4 = keras.layers.Multiply()([my_input8, COST_P])

    concat4 = keras.layers.concatenate([Mulp_3, Mulp_4])

    poids_setup7=np.array([[1],[1]])
    biais_setup7=np.array([0])
    COST = keras.layers.Dense(1, activation='linear', name="COST", weights=[poids_setup7, biais_setup7], trainable=False)(concat4)
    
    #---------------------------------------------------------------------------------------------------
	#creation du modele a partir des couches : definir l'entree et la sortie du reseau
    in_all = [my_input1, my_input2, my_input3, my_input4, my_input5, my_input6, my_input7, my_input8, my_input9]
    in_all += coucheInput_cost_P
    model = keras.Model(inputs=in_all, outputs=COST)
    
	#résumé du modèle
    model.summary()
    
	#dessin du modele
    keras.utils.plot_model(model, "PROD_reseauALternativeLearning.png", show_shapes=True, dpi=300)#192
    
    model.compile(optimizer='adam', loss='mean_squared_error')#Lorsqu'on mets le gaps il influence négative la qualité de modèle contruit
    
    #---------------------------------------------------------------------------------------------------
    #entrainement 
    in_PR_A_RP0_A_RG_A2 = np.asarray(in_PR_A_RP0_A_RG_A[90:])
    in_A_PR_A_N2 = np.asarray(in_A_PR_A_N[90:])
    in_A_moins_A0_A_plus_A02 = np.asarray(in_A_moins_A0_A_plus_A0[90:])
    in_A_plus_A02 = np.asarray(in_A_plus_A0[90:])
    in_P_moins_P0_P_plus_P02 = np.asarray(in_P_moins_P0_P_plus_P0[90:])
    in_P_plus_P02 = np.asarray(in_P_plus_P0[90:])
    in_K2 = np.asarray(in_K[90:])
    in_mu2 = np.asarray(in_mu[90:])
    out_Cost_A_P2 = np.asarray(out_Cost_A_P[90:])#
    input00 = np.asarray(input0[90:])
    input11 = np.asarray(input1[90:])
    input22 = np.asarray(input2[90:])
    input33 = np.asarray(input3[90:])
    input44 = np.asarray(input4[90:])
    input55 = np.asarray(input5[90:])
    #out_Cost_A_P2 = out_Cost_A_P2.astype(float)

	#entrainement du modele
    history = model.fit({"in1": in_PR_A_RP0_A_RG_A2, "in2": in_A_PR_A_N2, "in3":in_A_moins_A0_A_plus_A02, "in4":in_A_plus_A02, "in5": in_P_moins_P0_P_plus_P02, "in6":in_P_plus_P02, "K":in_K2, "mu":in_mu2,"input55":input55,
                   "input0": input00,"input1": input11,"input2": input22,"input3": input33,"input4": input44,"input5": input55
                   }, 
                 out_Cost_A_P2,
                 validation_split=0.33, epochs=200, batch_size=32, verbose=0) #debug epochs=1
    
    
    # on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("erreur en fonction des époques")
    plt.ylabel('loss (carré moyen des erreurs)')
    plt.xlabel('nombre d\'époques')
    plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
    plt.savefig('PROD_loss_reseauALternativeLearning.pdf')
    plt.show()

    model.save('my_model_learn_prod.h5')  # creates a HDF5 file 'my_model.h5'
    #del model  # deletes the existing model
    
def Cost_value_predict(in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list ):

    model = load_model('my_model_learn_prod.h5')

    in_PR_A_RP0_A_RG_A2 = np.asarray(in_PR_A_RP0_A_RG_A[90:])
    in_A_PR_A_N2 = np.asarray(in_A_PR_A_N[90:])
    in_A_moins_A0_A_plus_A02 = np.asarray(in_A_moins_A0_A_plus_A0[90:])
    in_A_plus_A02 = np.asarray(in_A_plus_A0[90:])
    in_P_moins_P0_P_plus_P02 = np.asarray(in_P_moins_P0_P_plus_P0[90:])
    in_P_plus_P02 = np.asarray(in_P_plus_P0[90:])
    in_K2 = np.asarray(in_K[90:])
    in_mu2 = np.asarray(in_mu[90:])
    out_Cost_A_P2 = np.asarray(out_Cost_A_P[90:])#
    input00 = np.asarray(input0[90:])
    input11 = np.asarray(input1[90:])
    input22 = np.asarray(input2[90:])
    input33 = np.asarray(input3[90:])
    input44 = np.asarray(input4[90:])
    input55 = np.asarray(input5[90:])


    #Représentation graphique des données d'apprentissage
    #représentation graphique en une unique courbe
    y = model.predict({"in1": in_PR_A_RP0_A_RG_A2, "in2": in_A_PR_A_N2, "in3":in_A_moins_A0_A_plus_A02, "in4":in_A_plus_A02, "in5": in_P_moins_P0_P_plus_P02, "in6":in_P_plus_P02, "K":in_K2, "mu":in_mu2,"input55":input55,
                   "input0": input00,"input1": input11,"input2": input22,"input3": input33,"input4": input44,"input5": input55})
    for k in range(len(y)): 
        if(y[k]<0):
            print("stop :",k,"val", y[k])
  
    Nb_valtest=out_Cost_A_P2.shape[0]
    y_train_temp=out_Cost_A_P[90:]
    y_train_temp_tri=y_train_temp
    y_tri=y
    y_train_temp_tri, y_tri =zip(*sorted(zip(y_train_temp_tri, y_tri)))
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)

    y_train_temp_tri_by_100=[]#y_train_temp_tri_by_100 est la liste obtenues après transformation de la liste y_train_temp_tri en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        y_train_temp_tri_by_100.append(np.mean(y_train_temp_tri[100*k:100*k+100]))
    y_train_temp_tri_by_100.append(np.mean(y_train_temp_tri[5900:5909]))

    y_tri_by_100=[]#y_tri_by_100 est la liste obtenues après transformation de la liste y_tri en la regroupant par paquet de 100 
    #x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        y_tri_by_100.append(np.mean(y_tri[100*k:100*k+100]))
    y_tri_by_100.append(np.mean(y_tri[5900:5909]))

    plt.plot(x, y_tri_by_100, "r", label="Coût de production prédit par le réseau de neurone")
    plt.plot(x, y_train_temp_tri_by_100, "g", label="Coût de production optimal" )
    plt.ylabel('coût de production')
    plt.xlabel('numéro de l\'instance')
    plt.legend()
    plt.savefig('PROD_prediction_reseauALternativeLearning2_train_data.pdf')
    plt.show()

    Nb_valtest=len(out_Cost_A_P[90:])
    err=0
    k_itt=90
    for i in range(Nb_valtest):
        if(i!=5241&i!=5501&i!=5555):
            err = err + (abs(out_Cost_A_P[k_itt] - y[i]) * 100) / out_Cost_A_P[k_itt] # on calcule l'erreur realisee par le reseau de neurones
        k_itt=k_itt+1
    print("erreur train= " + str(err / Nb_valtest) + "%")



    #Représentation graphique des données de test
    in_PR_A_RP0_A_RG_A2 = np.asarray(in_PR_A_RP0_A_RG_A[0:90])
    in_A_PR_A_N2 = np.asarray(in_A_PR_A_N[0:90])
    in_A_moins_A0_A_plus_A02 = np.asarray(in_A_moins_A0_A_plus_A0[0:90])
    in_A_plus_A02 = np.asarray(in_A_plus_A0[0:90])
    in_P_moins_P0_P_plus_P02 = np.asarray(in_P_moins_P0_P_plus_P0[0:90])
    in_P_plus_P02 = np.asarray(in_P_plus_P0[0:90])
    in_K2 = np.asarray(in_K[0:90])
    in_mu2 = np.asarray(in_mu[0:90])
    input00_test = np.asarray(input0[0:90])
    input11_test = np.asarray(input1[0:90])
    input22_test = np.asarray(input2[0:90])
    input33_test = np.asarray(input3[0:90])
    input44_test = np.asarray(input4[0:90])
    input55_test = np.asarray(input5[0:90])

    y = model.predict({"in1": in_PR_A_RP0_A_RG_A2, "in2": in_A_PR_A_N2, "in3":in_A_moins_A0_A_plus_A02, "in4":in_A_plus_A02, "in5": in_P_moins_P0_P_plus_P02, "in6":in_P_plus_P02, "K":in_K2, "mu":in_mu2, "input55":input55_test,
                   "input0": input00_test,"input1": input11_test,"input2": input22_test,"input3": input33_test,"input4": input44_test,"input5": input55_test
                   }) #prédiction des valeurs lambda
    y_train_temp=np.asarray(out_Cost_A_P[0:90])
    y_train_temp_tri=y_train_temp
    y_tri=y
    y_train_temp_tri, y_tri =zip(*sorted(zip(y_train_temp_tri, y_tri)))
    Nb_valtest=90
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    plt.plot(x, y_tri, "r", label="Coût de production prédit par le réseau de neurone")
    plt.plot(x, y_train_temp_tri, "g", label="Coût de production optimal" )
    plt.ylabel('coût de production')
    plt.xlabel('numéro de l\'instance')
    plt.legend()
    plt.savefig('PROD_prediction_reseauALternativeLearning2_test_data.pdf')
    plt.show()
    
    Nb_valtest=len(out_Cost_A_P[0:90])
    err=0
    for i in range(Nb_valtest):
        err = err + (abs(out_Cost_A_P[i] - y[i]) * 100) / out_Cost_A_P[i] # on calcule l'erreur realisee par le reseau de neurones
    print("erreur test = " + str(err / Nb_valtest) + "%")

	# affichage de l'erreur
	#print(err / Nb_valtest)
	

    #affichage des poids des couches
    tau= affichePoids(model, "tau", False)
    beta= affichePoids(model, "beta", False)
    omega= affichePoids(model, "omega", False)
    alpha0= affichePoids(model, "alpha0", False)
    alpha1= affichePoids(model, "alpha1", False)
    alpha2= affichePoids(model, "alpha2", False)
    alpha3= affichePoids(model, "alpha3", False)
    alpha4= affichePoids(model, "alpha4", False)
    alpha5= affichePoids(model, "alpha5", False)
    sigma= affichePoids(model, "sigma", False)
    
    listt_param=[]
    listt_param.append(tau[0])
    listt_param.append(tau[1])
    listt_param.append(tau[2])
    listt_param.append(beta[0])
    listt_param.append(beta[1])
    listt_param.append(beta[2])
    listt_param.append(omega)
    listt_param.append(alpha0)
    listt_param.append(alpha1)
    listt_param.append(alpha2)
    listt_param.append(alpha3)
    listt_param.append(alpha4)
    listt_param.append(alpha5)
    listt_param.append(sigma)

    listt_param_tri=listt_param
    xx=np.linspace(1,14,14)
    annotations=["t_1","t_2","t_3","bet_1","bet_2","bet_0","omga","alph_1","alph_2","alph_3","alph_4","alph_5","alph_0","sigma"]
    plt.plot(xx, listt_param_tri, "ob", label="param_Predict")
    for i, label in enumerate(annotations):
        plt.annotate(label, (xx[i], listt_param_tri[i]))
    plt.legend()
    plt.savefig('list_param_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()
    #------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------tau

    y_train_pr_recalcul=np.asarray(out_Cost_A_P)
    Nb_valtest=y_train_pr_recalcul.shape[0]
    x = np.linspace(1,Nb_valtest,Nb_valtest)

    # on recupère la couche interne à partir de son nom
    couche_tau = keras.Model(inputs=model.input, outputs=model.get_layer("tau").output)
    # on donne les inputs souhaités
    y_tau = couche_tau({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu), "input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })
    
    #couche_2AA = keras.Model(inputs=model.input, outputs=model.get_layer("couche2_A").output)
    #y_tau_et_1 = couche_2AA({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),"input55":np.asarray(input5),
    #               "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
    #               })
       

 
    #y_recalcul contient les valeurs tau recalculées en utilisant les valeurs de tau_i et de omega apprisent par le réseau
    tau_recalcule=[]
    for i in range(0,Nb_valtest):
       tau00=tau[0]*in_PR_A_RP0_A_RG_A[i][0]
       tau11=tau[1]*in_PR_A_RP0_A_RG_A[i][1]
       tau22=tau[2]*in_PR_A_RP0_A_RG_A[i][2]
       #print("tau :", tau00, tau11, tau22)
       tau_recalcule.append(tau[0]*in_PR_A_RP0_A_RG_A[i][0]+tau[1]*in_PR_A_RP0_A_RG_A[i][1]+tau[2]*in_PR_A_RP0_A_RG_A[i][2])
       #print("tau_0",tau[0])
       #print("tau_1",tau[1])
       #print("tau_2",tau[2])
       #print("Fin")
    tau_recalcule_tri=tau_recalcule
    y_tau_tri=y_tau
    tau_recalcule_tri, y_tau_tri =zip(*sorted(zip(tau_recalcule_tri, y_tau_tri)))
    plt.plot(x, y_tau_tri, "r", label="tau_Predict")
    plt.plot(x, tau_recalcule_tri, "y", label="tau_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('tau_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()

    #-------------------------------------------------------------------beta
    
    # on recupère la couche interne à partir de son nom (on veut la couche qui nous permet d'obtenir beta
    couche_beta = keras.Model(inputs=model.input, outputs=model.get_layer("omega").output)
    # on donne les inputs souhaités
    y_beta = couche_beta({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),"input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })
    #WS=model.get_layer("beta").get_weights()
    #on veut WS[1] qui est le biais

    #y_recalcul contient les valeurs beta recalculées en utilisant les valeurs de beta_i et de omega apprisent par le réseau
    beta_recalcule=[]
    for i in range(0,Nb_valtest):
        Y=beta[0]*in_A_PR_A_N[i][0]+beta[1]*in_A_PR_A_N[i][1]-beta[2]#on ne sait pas comment imposer que le biais WS[1] negatif
        beta_recalcule.append([exp(omega*Y)/(exp(omega*Y)+1)])
    beta_recalcule_tri=beta_recalcule
    y_beta_tri=y_beta
    beta_recalcule_tri, y_beta_tri =zip(*sorted(zip(beta_recalcule_tri, y_beta_tri)))
    plt.plot(x, y_beta_tri, "r", label="beta_Predict")
    plt.plot(x, beta_recalcule_tri, "y", label="beta_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('beta_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()


    couche_bet = keras.Model(inputs=model.input, outputs=model.get_layer("beta").output)
    # on donne les inputs souhaités
    y_Y = couche_bet({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),"input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })
    
    y_Y_tri=y_Y
    plt.plot(x, y_Y_tri, "ob", label="Y_Predict")
    plt.legend()
    plt.savefig('Y_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()
    #-------------------------------------------------------------------alpha
    #input0 = P0/N
    #input1 = A/P*R
    #input2 = G/P
    #input3 = V0/V
    #input4 = C
    #input5 = 1 
    
    couche_Z = keras.Model(inputs=model.input, outputs=model.get_layer("Z").output)#la couche sigma est celle qui nous donne alpha
    # on donne les inputs souhaités
    y_Z = couche_Z({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),"input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })
    # on recupère la couche interne à partir de son nom
    couche_alpha = keras.Model(inputs=model.input, outputs=model.get_layer("sigma").output)#la couche sigma est celle qui nous donne alpha
    # on donne les inputs souhaités
    y_alpha = couche_alpha({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),"input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })


    #y_recalcul contient les valeurs alpha recalculées en utilisant les valeurs de alpha_i et de omega apprisent par le réseau
    alpha_recalcule=[]
    for i in range(0,Nb_valtest):
        ZZ=alpha0*input0[i]-alpha1*input1[i]-alpha2*input2[i]-alpha3*input3[i]-alpha4*input4[i]-alpha5*input5[i]
        alpha_recalcule.append([exp(sigma*ZZ)/(exp(sigma*ZZ)+1)])
    alpha_recalcule_tri=alpha_recalcule
    y_alpha_tri=y_alpha
    alpha_recalcule_tri, y_alpha_tri =zip(*sorted(zip(alpha_recalcule_tri, y_alpha_tri)))
    plt.plot(x, y_alpha_tri, "r", label="alpha_Predict")
    plt.plot(x, alpha_recalcule_tri, "y", label="alpha_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('alpha_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()

    y_Z_tri=y_Z
    plt.plot(x, y_Z_tri, "ob", label="Z_Predict")
    plt.legend()
    plt.savefig('Z_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()
    #-------------------------------------------------------------------COST
    
    # on recupère la couche interne à partir de son nom
    couche_COST = keras.Model(inputs=model.input, outputs=model.get_layer("COST").output)
    y_cost = couche_COST({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu), "input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })
   
    couche_COST_P = keras.Model(inputs=model.input, outputs=model.get_layer("couche6_B").output)
    y_cost_p = couche_COST_P({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),  "input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })

    couche_COST_A_n = keras.Model(inputs=model.input, outputs=model.get_layer("COST_A_n").output)
    y_cost_A_n = couche_COST_A_n({"in1": np.asarray(in_PR_A_RP0_A_RG_A), "in2": np.asarray(in_A_PR_A_N), "in3":np.asarray(in_A_moins_A0_A_plus_A0), "in4":np.asarray(in_A_plus_A0), "in5": np.asarray(in_P_moins_P0_P_plus_P0), "in6":np.asarray(in_P_plus_P0), "K":np.asarray(in_K), "mu":np.asarray(in_mu),  "input55":np.asarray(input5),
                   "input0": np.asarray(input0),"input1": np.asarray(input1),"input2": np.asarray(input2),"input3": np.asarray(input3),"input4": np.asarray(input4),"input5": np.asarray(input5)
                   })
   
 
    print(len(in_PR_A_RP0_A_RG_A))
    COST_recalcule=[]
    for i in range(0,Nb_valtest):
        #COSTA=(1+tau_recalcule[i])*(beta_recalcule[i][0]*in_A_moins_A0_A_plus_A0[i][0] - beta_recalcule[i][0]*in_A_moins_A0_A_plus_A0[i][1]+ in_A_plus_A0[i][0])
        #COSTP=1*alpha_recalcule[i][0]*in_P_moins_P0_P_plus_P0[i][0] -1* alpha_recalcule[i][0]*in_P_moins_P0_P_plus_P0[i][1]+ 1*in_P_plus_P0[i][0]
        #COSTP=round(COSTP, 7)
        COSTA=(1+y_tau[i])*(y_beta[i]*in_A_moins_A0_A_plus_A0[i][0] - y_beta[i]*in_A_moins_A0_A_plus_A0[i][1]+ in_A_plus_A0[i][0])
        COSTP=y_alpha[i]*in_P_moins_P0_P_plus_P0[i][0] - y_alpha[i]*in_P_moins_P0_P_plus_P0[i][1]+ in_P_plus_P0[i][0]
        #COSTP=round(COSTP, 7)
        me=y_cost_p[i]
        if(y_cost_p[i]!=COSTP):
            print("!!!!!!!!!!!!!STOP")
        #if(y_cost_A_n[i]!=COSTA):
        #    print("!!!!!!!!!!!!!STOP_COSTA")
        COST_recalcule.append(in_K[i][0]*COSTA + in_mu[i][0]*COSTP)
        if(y_cost[i]!=COST_recalcule[i]):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!STOP")
    COST_recalcule_tri=COST_recalcule
    y_cost_tri=y_cost
    COST_recalcule_tri, y_cost_tri =zip(*sorted(zip(COST_recalcule_tri, y_cost_tri)))
    plt.plot(x, y_cost_tri, "r", label="COST_Predict")
    plt.plot(x, COST_recalcule_tri, "y", label="COST_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('COST_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()

    y_cost_p_tri=y_cost_p
    plt.plot(x, y_cost_p_tri, "ob", label="COST_P_Predict")
    plt.legend()
    plt.savefig('COSTP_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()

    y_cost_A_n_tri=y_cost_A_n
    plt.plot(x, y_cost_A_n_tri, "ob", label="COST_A_Predict")
    plt.legend()
    plt.savefig('COSTA_prediction_reseauALternativeLearning_train_and_test_data.png')
    plt.show()
    
    #création des fichiers beta_COSTA.txt et beta_COSTP.txt
    cheminComplet1 = "beta_COSTA.txt"
    cheminComplet2 = "beta_COSTP.txt"
    cheminComplet3 = "beta.txt"#COSTA/CMP + COSTP
    with open(cheminComplet1, "a") as fichier_RS:
        for i in range(21):
            np.savetxt(fichier_RS, y_cost_A_n[i], fmt="%.25f")
    with open(cheminComplet2, "a") as fichier_RSS:
        for i in range(21):
            np.savetxt(fichier_RSS, y_cost_p[i], fmt="%.25f")
    with open(cheminComplet3, "a") as fichier_RSSS:
        for i in range(21):
            np.savetxt(fichier_RSSS, (y_cost_A_n[i]/CMP_list[i])+y_cost_p[i], fmt="%.25f")


def Time_value2(input0, input1, input2, input3, input4, input5, input6, input7, output_period_last_recharge, recalcul_T_MinQ, recalcul_T_MaxQ, recalcul_T_Delta, output):
    #---------------------------------------------------------------------------------------------------
    #ETAPE 1. on va definir chaque neurones de la couche input separement, on les stocke dans un tableau
    my_input1 = keras.Input(shape=(8,), name="input1")
    out_val1 = keras.layers.Dense(1, activation='sigmoid', name="TV_lambda", kernel_constraint=keras.constraints.NonNeg(), bias_constraint=keras.constraints.NonNeg())(my_input1)
    out_val = keras.layers.Dense(1, activation='sigmoid', name="TV_gamma", kernel_constraint=keras.constraints.NonNeg(), use_bias=False)(out_val1)
    #

    #---------------------------------------------------------------------------------------------------
	# ETAPE 2.  creation du modele a partir des couches : definir l'entree et la sortie du reseau
    model = keras.Model(inputs=my_input1, outputs=out_val)
    
	#résumé du modèle
    model.summary()
    
	#dessin du modele
    keras.utils.plot_model(model, "reseauALternativeLearning2.png", show_shapes=True, dpi=192)

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    #---------------------------------------------------------------------------------------------------
	# ETAPE 3.  entrainement du modele

    my_input_pred=[]
    y_train2 = np.asarray(output[90:])#
    
    for j in range(0,y_train2.shape[0]+90):
        input5[j][0]=-1*input5[j][0]
        input6[j][0]=-1*input6[j][0]
        input7[j][0]=-1*input7[j][0]
    for j in range(90,y_train2.shape[0]+90):##données d'entrainements sans les données de test 90,y_train2.shape[0]+90.............0,y_train2.shape[0]
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input = np.asarray(my_input_pred)

	#entrainement
    history = model.fit(my_input, y_train2, validation_split=0.33, epochs=200, batch_size=32, verbose=0)
    
    # on peut afficher l'évolution de l'erreur en fonction du nombre d'epochs
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("erreur en fonction des époques")
    plt.ylabel('loss (carré moyen des erreurs)')
    plt.xlabel('nombre d\'époques')
    plt.legend(['erreur sur les données d\'apprentissage', 'erreur sur les données de test'], loc='upper left')
    plt.savefig('loss_reseauALternativeLearning2.png')
    plt.show()

    #Représentation graphique des données d'apprentissage
    #représentation graphique en une unique courbe
    y = model.predict(my_input)
    Nb_valtest=y_train2.shape[0]
    for i in range(90,Nb_valtest):
        y[i][0]=y[i][0]*(recalcul_T_MinQ[i][0] + recalcul_T_Delta[i][0]) +(1-y[i][0])*recalcul_T_MaxQ[i]
    y_train_temp=output_period_last_recharge[90:]
    y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    plt.plot(x, y_train_temp, "g", label="Opt" )
    plt.plot(x, y, "r", label="Predict")
    plt.legend()
    plt.savefig('prediction_reseauALternativeLearning2_train_data.png')
    plt.show()


    #Représentation graphique des données de test
    my_input_pred=[]
    y_train2 = np.asarray(output[0:90])#
    
    for j in range(0,90):##données d'entrainements sans les données de test 90,y_train2.shape[0]+90.............0,y_train2.shape[0]
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input_test = np.asarray(my_input_pred)

    y = model.predict(my_input_test) #prédiction des valeurs lambda
    for i in range(0,90):
        y[i][0]=y[i][0]*(recalcul_T_MinQ[i][0] + recalcul_T_Delta[i][0]) +(1-y[i][0])*recalcul_T_MaxQ[i]
    y_train_temp=output_period_last_recharge[0:90]
    y_train_temp, y =zip(*sorted(zip(y_train_temp, y)))
    Nb_valtest=90
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    plt.plot(x, y_train_temp, "g", label="Opt" )
    plt.plot(x, y, "r", label="Predict")
    plt.legend()
    plt.savefig('prediction_reseauALternativeLearning2_test_data.png')
    plt.show()

    #affichage des poids des couches
    TV_lamb=affichePoids(model, "TV_lambda", True)
    TV_gamm=affichePoids(model, "TV_gamma", False)

     #------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------lambda
    #courbe du lambda opt, du lambda prédit et du lambda recalculé
    y_train_pr_recalcul = np.asarray(output)#

    Nb_valtest=y_train_pr_recalcul.shape[0]
    my_input_pred=[]
    for j in range(0,Nb_valtest):
        myinput_ligne=input0[j]+input1[j]+input2[j]+input3[j]+input4[j]+input5[j]+input6[j]+input7[j]
        my_input_pred.append(myinput_ligne)
    my_input = np.asarray(my_input_pred)
        
    y = model.predict(my_input)

    #y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    y_recalcule=[]
    for i in range(0,Nb_valtest):
        X=TV_lamb[0]*input0[i]+TV_lamb[1]*input1[i]+TV_lamb[2]*input2[i]+TV_lamb[3]*input3[i]+TV_lamb[4]*input4[i]+TV_lamb[5]*input5[i]+TV_lamb[6]*input6[i]+TV_lamb[7]*input7[i]
        y_recalcule.append([exp(TV_gamm*X)/(exp(TV_gamm*X)+1)])
    
    y_train_temp_out=output
    y_1=y
    y_2=y
    y_train_temp_out, y_1 =zip(*sorted(zip(y_train_temp_out, y_1)))
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    plt.plot(x, y_train_temp_out, "g", label="lambda_Opt" )
    plt.plot(x, y_1, "r", label="lambda_Predict")
    plt.legend()
    plt.savefig('lambda_prediction_reseauALternativeLearning_train_and_test_data2.png')
    plt.show()

    y_recalcule, y_2 =zip(*sorted(zip(y_recalcule, y_2)))
    plt.plot(x, y_2, "r", label="lambda_Predict")
    plt.plot(x, y_recalcule, "y", label="lambda_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('lambda2_prediction_reseauALternativeLearning_train_and_test_data2.png')
    plt.show()

    #---------------------------------------------------------------------T
     #courbe du T opt, du T prédit et du T recalculé

    T_recalcule=[]
    for i in range(0,Nb_valtest):
        T_recalcule.append([(y[i][0]*(recalcul_T_MinQ[i][0]+recalcul_T_Delta[i][0]))+((1-y[i][0])*recalcul_T_MaxQ[i])])

    for i in range(0,Nb_valtest):
        y[i][0]=y[i][0]*(recalcul_T_MinQ[i][0] + recalcul_T_Delta[i][0]) +(1-y[i][0])*recalcul_T_MaxQ[i]
    y_train_temp=output_period_last_recharge
    y_1=y
    y_2=y

    y_train_temp, y_1 =zip(*sorted(zip(y_train_temp, y_1)))
    x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    plt.plot(x, y_train_temp, "g", label="T_Opt" )
    plt.plot(x, y_1, "r", label="T_Predict")
    plt.legend()
    plt.savefig('T_prediction_reseauALternativeLearning_train_and_test_data2.png')
    plt.show()

    T_recalcule, y_2 =zip(*sorted(zip(T_recalcule, y_2)))
    plt.plot(x, y_2, "r", label="T_Predict")
    plt.plot(x, T_recalcule, "y", label="T_recalculé")#y_recalcul contient les valeurs lambda recalculées en utilisant les valeurs de lambda_i et de gamma apprisent par le réseau
    plt.legend()
    plt.savefig('T2_prediction_reseauALternativeLearning_train_and_test_data2.png')
    plt.show()


#Cette fonction sert à exécuter Cost_value_model_ une certains nombre de fois égale au nombre d'epochs voulu.
# On le fait juste pour pouvoir calculer les gaps des données de test et d'entrainement à chaque epochs pour déssiner une courbe de l'évolution des gaps au fil des epochs
def Cost_value_model_fig_gap_by_epoch(Nb_epochs, in_PR_A_RP0_A_RG_A, in_A_PR_A_N, in_A_moins_A0_A_plus_A0, in_A_plus_A0, in_P_moins_P0_P_plus_P0, in_P_plus_P0, in_K, in_mu, out_Cost_A_P, input0, input1, input2, input3, input4, input5, CMP_list ):
    #debug initializer = tf.keras.initializers.RandomUniform(minval=1, maxval=5)
    
     #---------------------------------Partie du code qui évalue COST_A--------------------------------------------------------------

    my_input1 = keras.Input(shape=(3,), name="in1")#PR_A_RP0_A_RG_A
    my_input2 = keras.Input(shape=(3,), name="in2")#in_A_PR_A_N
    my_input3 = keras.Input(shape=(2,), name="in3")#A_moins_A0_A_plus_A0
    my_input4 = keras.Input(shape=(1,), name="in4")#A_plus_A0
    my_input5 = keras.Input(shape=(2,), name="in5")#P_moins_P0_P_plus_P0
    my_input6 = keras.Input(shape=(1,), name="in6")#P_plus_P0
    my_input7 = keras.Input(shape=(1,), name="K")#corresponds à K et mu des indicateurs
    my_input8 = keras.Input(shape=(1,), name="mu")
    my_input9 = keras.Input(shape=(1,), name="input55")
    #paramètres : tau_1, tau_2, tau_3
    #sortie : tau
    couche1_A = keras.layers.Dense(1, activation='linear', name="tau", use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomUniform(minval=3, maxval=4)
                                   )(my_input1)
    poids_setup1=np.array([[1],[1]])
    biais_setup1=np.array([0])
    #sortie : tau+1
    concat0 = keras.layers.concatenate([couche1_A, my_input9])
    couche2_A = keras.layers.Dense(1, activation='linear', name="couche2_A", weights=[poids_setup1, biais_setup1], trainable=False)(concat0)
    
    #paramètres : beta_1, beta_2, beta_0
    #sortie : Y
    couche1_B = keras.layers.Dense(1, activation='linear', name="beta", kernel_constraint=keras.constraints.NonNeg(), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=1/2, maxval=3)
                                   )(my_input2)
    
    #paramètre : omega
    #sortie : beta
    couche2_B =  keras.layers.Dense(1, activation='sigmoid', name="omega", use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=1)
                                    )(couche1_B)
    
    poids_setup2=np.array([[1]])
    biais_setup2=np.array([0])
    #paramètres : 1
    #sortie : beta
    couche3_B_1 = keras.layers.Dense(1, activation='linear', name="couche3_B_1", weights=[poids_setup2, biais_setup2], trainable=False)(couche2_B)
    #paramètres : 1
    #sortie : beta
    couche3_B_2 = keras.layers.Dense(1, activation='linear', name="couche3_B_2", weights=[poids_setup2, biais_setup2], trainable=False)(couche2_B)
    couche3_B = keras.layers.concatenate([couche3_B_1, couche3_B_2])

    Mulp_1 = keras.layers.Multiply()([my_input3, couche3_B])
    
    concat1 = keras.layers.concatenate([Mulp_1, my_input4])

    poids_setup3=np.array([[1],[-1],[1]])
    biais_setup3=np.array([0])
    couche4_B = keras.layers.Dense(1, activation='linear', name="couche4_B", weights=[poids_setup3, biais_setup3], trainable=False)(concat1)
    
    COST_AA = keras.layers.Multiply()([couche2_A, couche4_B])

    poids_setup2=np.array([[1]])
    biais_setup2=np.array([0])
    #paramètres : 1
    #sortie : COST_A
    COST_A = keras.layers.Dense(1, activation='linear', name="COST_A_n", weights=[poids_setup2, biais_setup2], trainable=False)(COST_AA)
    
    #---------------------------------Partie du code qui évalue COST_P--------------------------------------------------------------

    coucheInput_cost_P = []
    #input0 = P0/N
    #input1 = A/P*R
    #input2 = G/P
    #input3 = V0/V
    #input4 = C
    #input5 = 1 
    for i in range(6):
        input_tmp = keras.Input(shape=(1,), name="input"+str(i)) 
        coucheInput_cost_P.append(input_tmp)

    couche1_A_cost_P=[]
    #coucheInput_cost_P[0] ------ l'input de cette couche est le vecteur de P0/N correspondant à alpha_0 ---- alpha1 de l'article
    #coucheInput_cost_P[1] ------ l'input de cette couche est le vecteur de A/P*R correspondant à alpha_1 ---- alpha2 de l'article
    #coucheInput_cost_P[2] ------ l'input de cette couche est le vecteur de G/P correspondant à alpha_2 ---- alpha3 de l'article
    #coucheInput_cost_P[3] ------ l'input de cette couche est le vecteur de V0/V correspondant à alpha_3 ---- alpha4 de l'article
    #coucheInput_cost_P[4] ------ l'input de cette couche est le  de C correspondant à alpha_4 ---- alpha5 de l'article
    #coucheInput_cost_P[5] ------ l'input de cette couche est le vecteur de 1 correspondant à alpha_5 ---- alpha0 de l'article
    for i in range(5):
        tmp = keras.layers.Dense(1, activation='linear',name="alpha"+str(i), use_bias=False, kernel_constraint=keras.constraints.NonNeg(), kernel_initializer=tf.keras.initializers.RandomUniform(minval=1/2, maxval=1)
                                 )(coucheInput_cost_P[i])
        couche1_A_cost_P.append(tmp)
    tmp = keras.layers.Dense(1, activation='linear',name="alpha"+str(5), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0)
                             )(coucheInput_cost_P[5])
    couche1_A_cost_P.append(tmp)
    concat2 = keras.layers.concatenate(couche1_A_cost_P)

    #le dernier parmètre est le billet
    poids_setup4=np.array([[1],[-1],[-1],[-1],[-1],[-1]])
    biais_setup4=np.array([0])
    #sortie : Z
    Z = keras.layers.Dense(1, activation='linear', name="Z", weights=[poids_setup4, biais_setup4], trainable=False)(concat2)
    #paramètres: sigma
    #sortie : alpha
    alpha = keras.layers.Dense(1, activation='sigmoid', name="sigma", kernel_constraint=keras.constraints.NonNeg(), use_bias=False, kernel_initializer=tf.keras.initializers.RandomUniform(minval=1, maxval=1)
                               )(Z)

    poids_setup5=np.array([[1]])
    biais_setup5=np.array([0])
    #sortie : alpha
    couche5_B_1 = keras.layers.Dense(1, activation='linear', name="double_alpha_1", weights=[poids_setup5, biais_setup5], trainable=False)(alpha)
    #sortie : alpha
    couche5_B_2 = keras.layers.Dense(1, activation='linear', name="double_alpha_2", weights=[poids_setup5, biais_setup5], trainable=False)(alpha)
    couche5_B = keras.layers.concatenate([couche5_B_1, couche5_B_2])

    Mulp_2 = keras.layers.Multiply()([my_input5, couche5_B])
    
    concat3 = keras.layers.concatenate([Mulp_2, my_input6])

    poids_setup6=np.array([[1],[-1],[1]])
    biais_setup6=np.array([0])
    COST_P = keras.layers.Dense(1, activation='linear', name="couche6_B", weights=[poids_setup6, biais_setup6], trainable=False)(concat3)
    
    #---------------------------------Partie du code qui évalue COST--------------------------------------------------------------

    Mulp_3 = keras.layers.Multiply()([my_input7, COST_A])
    Mulp_4 = keras.layers.Multiply()([my_input8, COST_P])

    concat4 = keras.layers.concatenate([Mulp_3, Mulp_4])

    poids_setup7=np.array([[1],[1]])
    biais_setup7=np.array([0])
    COST = keras.layers.Dense(1, activation='linear', name="COST", weights=[poids_setup7, biais_setup7], trainable=False)(concat4)
    
    #---------------------------------------------------------------------------------------------------
	#creation du modele a partir des couches : definir l'entree et la sortie du reseau
    in_all = [my_input1, my_input2, my_input3, my_input4, my_input5, my_input6, my_input7, my_input8, my_input9]
    in_all += coucheInput_cost_P
    model = keras.Model(inputs=in_all, outputs=COST)
    
	#résumé du modèle
    #model.summary()
    
	#dessin du modele
    keras.utils.plot_model(model, "PROD_reseauALternativeLearning.png", show_shapes=True, dpi=300)#192
    
    model.compile(optimizer='adam', loss='mean_squared_error')#Lorsqu'on mets le gaps il influence négative la qualité de modèle contruit
    
    #---------------------------------------------------------------------------------------------------
    #entrainement 
    in_PR_A_RP0_A_RG_A2 = np.asarray(in_PR_A_RP0_A_RG_A[90:])
    in_A_PR_A_N2 = np.asarray(in_A_PR_A_N[90:])
    in_A_moins_A0_A_plus_A02 = np.asarray(in_A_moins_A0_A_plus_A0[90:])
    in_A_plus_A02 = np.asarray(in_A_plus_A0[90:])
    in_P_moins_P0_P_plus_P02 = np.asarray(in_P_moins_P0_P_plus_P0[90:])
    in_P_plus_P02 = np.asarray(in_P_plus_P0[90:])
    in_K2 = np.asarray(in_K[90:])
    in_mu2 = np.asarray(in_mu[90:])
    out_Cost_A_P2 = np.asarray(out_Cost_A_P[90:])#
    input00 = np.asarray(input0[90:])
    input11 = np.asarray(input1[90:])
    input22 = np.asarray(input2[90:])
    input33 = np.asarray(input3[90:])
    input44 = np.asarray(input4[90:])
    input55 = np.asarray(input5[90:])
    #out_Cost_A_P2 = out_Cost_A_P2.astype(float)

	#entrainement du modele
    history = model.fit({"in1": in_PR_A_RP0_A_RG_A2, "in2": in_A_PR_A_N2, "in3":in_A_moins_A0_A_plus_A02, "in4":in_A_plus_A02, "in5": in_P_moins_P0_P_plus_P02, "in6":in_P_plus_P02, "K":in_K2, "mu":in_mu2,"input55":input55,
                   "input0": input00,"input1": input11,"input2": input22,"input3": input33,"input4": input44,"input5": input55
                   }, 
                 out_Cost_A_P2,
                 validation_split=0.33, epochs=Nb_epochs, batch_size=32, verbose=0) #debug epochs=1


    in_PR_A_RP0_A_RG_A2 = np.asarray(in_PR_A_RP0_A_RG_A[90:])
    in_A_PR_A_N2 = np.asarray(in_A_PR_A_N[90:])
    in_A_moins_A0_A_plus_A02 = np.asarray(in_A_moins_A0_A_plus_A0[90:])
    in_A_plus_A02 = np.asarray(in_A_plus_A0[90:])
    in_P_moins_P0_P_plus_P02 = np.asarray(in_P_moins_P0_P_plus_P0[90:])
    in_P_plus_P02 = np.asarray(in_P_plus_P0[90:])
    in_K2 = np.asarray(in_K[90:])
    in_mu2 = np.asarray(in_mu[90:])
    out_Cost_A_P2 = np.asarray(out_Cost_A_P[90:])#
    input00 = np.asarray(input0[90:])
    input11 = np.asarray(input1[90:])
    input22 = np.asarray(input2[90:])
    input33 = np.asarray(input3[90:])
    input44 = np.asarray(input4[90:])
    input55 = np.asarray(input5[90:])


    #Représentation graphique des données d'apprentissage
    #représentation graphique en une unique courbe
    y = model.predict({"in1": in_PR_A_RP0_A_RG_A2, "in2": in_A_PR_A_N2, "in3":in_A_moins_A0_A_plus_A02, "in4":in_A_plus_A02, "in5": in_P_moins_P0_P_plus_P02, "in6":in_P_plus_P02, "K":in_K2, "mu":in_mu2,"input55":input55,
                   "input0": input00,"input1": input11,"input2": input22,"input3": input33,"input4": input44,"input5": input55})

    Nb_valtest=len(out_Cost_A_P[90:])
    Nb_valtest_train=len(out_Cost_A_P[90:])
    err_train=0
    k_itt=90
    for i in range(Nb_valtest):
        err_train = err_train + (abs(out_Cost_A_P[k_itt] - y[i]) * 100) / out_Cost_A_P[k_itt] # on calcule l'erreur realisee par le reseau de neurones
        k_itt=k_itt+1
    print("erreur train= " + str(err_train / Nb_valtest) + "%")



    #Représentation graphique des données de test
    in_PR_A_RP0_A_RG_A2 = np.asarray(in_PR_A_RP0_A_RG_A[0:90])
    in_A_PR_A_N2 = np.asarray(in_A_PR_A_N[0:90])
    in_A_moins_A0_A_plus_A02 = np.asarray(in_A_moins_A0_A_plus_A0[0:90])
    in_A_plus_A02 = np.asarray(in_A_plus_A0[0:90])
    in_P_moins_P0_P_plus_P02 = np.asarray(in_P_moins_P0_P_plus_P0[0:90])
    in_P_plus_P02 = np.asarray(in_P_plus_P0[0:90])
    in_K2 = np.asarray(in_K[0:90])
    in_mu2 = np.asarray(in_mu[0:90])
    input00_test = np.asarray(input0[0:90])
    input11_test = np.asarray(input1[0:90])
    input22_test = np.asarray(input2[0:90])
    input33_test = np.asarray(input3[0:90])
    input44_test = np.asarray(input4[0:90])
    input55_test = np.asarray(input5[0:90])

    y = model.predict({"in1": in_PR_A_RP0_A_RG_A2, "in2": in_A_PR_A_N2, "in3":in_A_moins_A0_A_plus_A02, "in4":in_A_plus_A02, "in5": in_P_moins_P0_P_plus_P02, "in6":in_P_plus_P02, "K":in_K2, "mu":in_mu2, "input55":input55_test,
                   "input0": input00_test,"input1": input11_test,"input2": input22_test,"input3": input33_test,"input4": input44_test,"input5": input55_test
                   }) #prédiction des valeurs lambda
    
    Nb_valtest=len(out_Cost_A_P[0:90])
    err_test=0
    for i in range(Nb_valtest):
        err_test = err_test + (abs(out_Cost_A_P[i] - y[i]) * 100) / out_Cost_A_P[i] # on calcule l'erreur realisee par le reseau de neurones
    print("erreur test = " + str(err_test / Nb_valtest) + "%")

    return err_test/ Nb_valtest, err_train/ Nb_valtest_train
	
