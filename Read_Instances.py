from collections import Counter
import matplotlib.pyplot as plt
import string
import numpy as np
import math
DEBUG = False
# Fonction qui transforme le fichier en un seul vecteur dont les lignes sont placees
# les unes à la suite des autres
def Read_Sequential(chemin, num_file):
    ch_num_file = "%d" % num_file # Transforme num_file en chaine de caractere
    cheminComplet = chemin + ch_num_file + ".txt"
    with open(cheminComplet, "r") as fichier_RS:
        lines = fichier_RS.readlines() # Lecture de toutes les lignes du fichier
        if DEBUG:
            print(lines)
        return lines

# Fonction qui transforme le fichier en plusieurs vecteurs dont chaque ligne du fichier est un vecteur
def Read_Line_By_Line(chemin, num_file):
    ch_num_file = "%d" % num_file # Transforme num_file en chaine de caractere
    cheminComplet = chemin + ch_num_file + ".txt"
    with open(cheminComplet, "r") as fichier_RS:
        rend = fichier_RS.readline() # Lecture d'une ligne du fichier
        CoutFixe = fichier_RS.readline()
        CoutVar = fichier_RS.readline()
        CoutProd = fichier_RS.readline()
        DateDebLastRecharg = fichier_RS.readline()
        QteRecharg = fichier_RS.readline()
        if DEBUG:
            print(rend, CoutFixe, CoutVar, CoutProd, DateDebLastRecharg, QteRecharg)
        return rend, CoutFixe, CoutVar, CoutProd, DateDebLastRecharg, QteRecharg

# Fonction qui transforme le fichier en vecteur "par type" : 
# 36 10 10 ...10 1 ..1 1 .. 1 0.. 12 .. 22
def Input_Output_RN_By_Type(chemin, num_file):
    x_data = [] # Un vecteur contenant toutes les donnees d'une instance
    y_data = []
    rend, CoutFixe, CoutVar, CoutProd, DateDebLastRecharg, QteRecharg = Read_Line_By_Line(chemin, num_file)
    x_data  = DateDebLastRecharg + rend + CoutFixe + CoutVar + QteRecharg
    x_data = x_data.split() # Conversion d'une chaine de caractère en liste dont le critere de separation l'espace
    for i in range(0, len(x_data)): # Conversion d'une liste de caractere en liste d'entier
        x_data[i] = int(x_data[i]) 
    if DEBUG:
        print("Les entrees du réseau de neurones :", x_data)
    y_data = int(CoutProd) # Sortie
    if DEBUG:
        print("Les sorties du réseau de neurones :", y_data)
    return x_data, y_data

# Fonction qui transforme le fichier en vecteur "periode par periode"
# 36 10 1 1 0 10 1 1 0 ...10 1 1 22 ... 10 1 1 0
def Input_Output_RN_Period_By_Period(chemin, num_file):
    x_data = [] # Un vecteur contenant toutes les donnees d'une instance
    y_data = []
    rend, CoutFixe, CoutVar, CoutProd, DateDebLastRecharg, QteRecharg = Read_Line_By_Line(chemin, num_file)
    rend = rend.split() # Conversion d'une chaine de caractère en liste dont le critere de separation l'espace
    CoutFixe = CoutFixe.split()
    CoutVar = CoutVar.split()
    QteRecharg = QteRecharg.split()
    x_data  = [DateDebLastRecharg, rend[0], CoutFixe[0], CoutVar[0], QteRecharg[0]]
    for i in range(0, len(rend)):
        x_data+=[rend[i], CoutFixe[i], CoutVar[i], QteRecharg[i]]
    for i in range(0, len(x_data)): # Conversion d'une liste de caractere en liste d'entier
        x_data[i] = int(x_data[i]) 
    if DEBUG:
        print("Les entrees du réseau de neurones :", x_data)
    y_data = int(CoutProd) # Sortie
    if DEBUG:
        print("Les sorties du réseau de neurones :", y_data)
    return x_data, y_data

#Fonction qui permet de construire les X_data et Y_data qu'utilisera le modèle Réseau de neurones
#min_num_inst, max_num_inst : interalle des numéros d'instances
# choix_type_lecture :  permet de savoir quelle fonction de transformaion de données utilisée
    #si choix_type_lecture = 0 alors on utilise la transformation en fichier en vecteur "par type"
    #si choix_type_lecture = 1 alors on utilise la transformation en fichier en vecteur "periode par periode
def X_data_Y_data_construct(min_num_inst, max_num_inst, choix_type_lecture,chemin):
    X_data = []
    Y_data = []
    if(choix_type_lecture==0):
        #fichier en vecteur "par type"
        for i in range(min_num_inst, max_num_inst):
            x_data, y_data = Input_Output_RN_By_Type(chemin, i)
            X_data.append(x_data)
            Y_data.append(y_data)
            if DEBUG:
                print("\n",i)
 
    if(choix_type_lecture==1):
        #fichier en vecteur "periode par periode
        for i in range(min_num_inst, max_num_inst):
            x_data, y_data = Input_Output_RN_Period_By_Period(chemin, i)
            X_data.append(x_data)
            Y_data.append(y_data)
            if DEBUG:
                print("\n",i)
    return X_data, Y_data   

def Input_Output_statistiques(chemin, max_num_inst, min_num_inst):
    M_list = [] # Un vecteur contenant toutes les donnees d'une instance
    N_list = []
    v0_list = []
    vmax_list = []
    p_list = []
    alpha_list = []
    beta_list = []
    cout_fix_list = []
    E0_list = []
    Ctank_list = []
    X=[]
    Y=[]
    X_N=[]
    Y_N=[]
    for i in range(min_num_inst, max_num_inst):
        ch_num_file = "%d" % i # Transforme num_file en chaine de caractere
        cheminComplet = chemin + ch_num_file + ".txt"
        with open(cheminComplet, "r") as fichier_RS:
            M_list.append(fichier_RS.readline())
            N_list.append(fichier_RS.readline())
            v0_list.append(fichier_RS.readline())
            vmax_list.append(fichier_RS.readline())
            p_list.append(fichier_RS.readline())
            alpha_list.append(fichier_RS.readline())
            beta_list.append(fichier_RS.readline())
            cout_fix_list.append(fichier_RS.readline())
            E0_list.append(fichier_RS.readline())
            Ctank_list.append(fichier_RS.readline())
    count_M = Counter(M_list).most_common() #compte le nombre d'occurence de chaque entier d'une liste
    #nombre d'instances pour chaque M
    for i in range(0,len(count_M)) :
        X.append(count_M[i][0])
        Y.append(count_M[i][1])
    for i in range(0,len(X)) :
        X[i] = int(X[i].rstrip('\n'))
    #print(X)
    #print(Y)
    plt.bar(X,Y)
    #plt.legend()
    plt.xlabel("Nombre de stations")
    plt.ylabel("Nombre d'instances")
    plt.savefig('CourbeM.pdf')
    plt.show()

    #nombre d'instances pour chaque N
    #print(N_list)
    #print(p_list)
    for i in range(0,len( p_list)) :
       p_list[i] = int(p_list[i].rstrip('\n'))
       N_list[i] = int(N_list[i].rstrip('\n'))
       N_list[i] = N_list[i]/p_list[i]
    count_N = Counter(N_list).most_common()
    #print(N_list)
    #print(count_N)
    for i in range(0,len(count_N)) :
        X_N.append(count_N[i][0])
        Y_N.append(count_N[i][1])   
    #print(X_N)
    #print(Y_N)
    plt.bar(X_N,Y_N)
    #plt.legend()
    plt.xlabel("Nombre de périodes")
    plt.ylabel("Nombre d'instances")
    plt.savefig('CourbeN.pdf')
    plt.show()
 
def Input_Output_CustomLoss(chemin_instance, chemin_prodRN, chemin_prodRN_suite, min_num_inst, max_num_inst,nature_ytrain):
    CV_train = []
    CF_train = []
    REnd_train = []
    Lambda_train = []
    Arange_train = []
    Y_train = []
    Y_LIST_int = []
    count_Nb_recharge = []
    Date_last_recharg = []
    N = 0
    N_limite = 200000#permet de compacter les instances et se limiter à N=20
    kkk = -1
    STATS_nb_station = []
    STATS_N = []#TMax
    STATS_CF = []
    STATS_vmax = []
    STATS_ctank = []
    STATS_cout_var_moyen = []
    STATS_N_Vr = []#N
    for i in range(min_num_inst, max_num_inst):
        #print(i)
        Cv_train = []
        Cf_train = []
        Rend_train = []
        lambda_train = []
        arange_train = []
        #y_train = []
        #Cv_train_one = []
       # Cf_train_one = []
        #Rend_train_one = []
        #lambda_train_one = []
        #arange_train_one = []
        #y_train_one = []
        QteRecharg_one_int = []
        y_list_int = []
        mm_list_int = []
        MM_list_int = []
        muS_list = []
        ch_num_file = "%d" % i # Transforme num_file en chaine de caractere
        cheminComplet = chemin_instance + ch_num_file + ".txt"
        with open(cheminComplet, "r") as fichier_RS:
            M_list = fichier_RS.readline()
            N_list = fichier_RS.readline()
            v0_list = fichier_RS.readline()
            vmax_list = fichier_RS.readline()
            p_list = fichier_RS.readline()
            alpha_list = fichier_RS.readline()
            beta_list = fichier_RS.readline()
            cout_fix_list = fichier_RS.readline()
            E0_list = fichier_RS.readline()
            Ctank_list = fichier_RS.readline()

#utiliser ce if si on veut sélectionner uuniquement les instances dont N est
#inférieure à une certaine valeur
        if(int(N_list.rstrip('\n')) // int(p_list.rstrip('\n')) <= N_limite):
            rend_one, CoutFixe_one, CoutVar_one, CoutProd_one, DateDebLastRecharg_one, QteRecharg_one = Read_Line_By_Line(chemin_prodRN, i)
            for j in CoutVar_one.split():
                Cv_train.append(int(j))
            for j in CoutFixe_one.split():
                Cf_train.append(int(j))
            for j in rend_one.split():
                Rend_train.append(int(j))
            alpha_list = int(alpha_list.rstrip('\n'))
            p_list = int(p_list.rstrip('\n'))
            DateDebLastRecharg_one = int(DateDebLastRecharg_one.rstrip('\n'))
            DateDebLastRecharg_one = DateDebLastRecharg_one // p_list
            CoutProd_one = int(CoutProd_one.rstrip('\n'))
            #Si le dernier élément de la fonction Input_Output_CustomLoss vaut
            #0 alors y_train=last_refuel+coutprod
            #Si le dernier élément de la fonction Input_Output_CustomLoss vaut
            #1 alors y_train=last_refuel
            #Si le dernier élément de la fonction Input_Output_CustomLoss vaut
            #2 alors y_train=coutprod
            if(nature_ytrain == 0):
                y_train = alpha_list * (DateDebLastRecharg_one) + CoutProd_one
            if(nature_ytrain == 1):
                y_train = alpha_list * (DateDebLastRecharg_one)#à modifier pour exécuter 6000 instances (enlever "//plist")
            if(nature_ytrain == 2):
                y_train = CoutProd_one
            N_actuel = int(N_list.rstrip('\n')) // p_list
            if(N < N_actuel):
                N = N_actuel
            QteRecharg_one = QteRecharg_one.rstrip('\n')
            #print(QteRecharg_one)
            mu = []
            mu_ = []#mu*
            F = []
            L = []
            deb = -1
            fin = -1
            for j in QteRecharg_one.split():
                QteRecharg_one_int.append(int(j))
            ch_num_file = "%d" % i # Transforme num_file en chaine de caractere
            cheminComplets = chemin_prodRN_suite + ch_num_file + ".txt"
            with open(cheminComplets, "r") as fichier_RS:
                #y_list = fichier_RS.readline() #désacticver ceci pour exécuter les 6000 instances #vecteur production qui correspond à y dans le modèle réseau de neurones
                y_list='0 0 0 0'#acticver ceci pour exécuter les 6000 instances
                Nb_recharge = fichier_RS.readline() #le nombre de recharge calculé par la partie véhicule du pipeline
                mm_list = fichier_RS.readline() #periode au plutot possible de chaque s ieme recharge
                MM_list = fichier_RS.readline() #periode au plustard possible de chaque s ieme recharge
                muS_list = fichier_RS.readline()#Quantite d'hydrogene recharger par le vehicule a chaque s ieme recharge
            for j in y_list.split():
                y_list_int.append(int(j)) #conversion en entier du vecteur production
            for j in mm_list.split():
                mm_list_int.append(int(j))
            for j in MM_list.split():
                MM_list_int.append(int(j))
            for j in muS_list.split():#remplisage de la liste mu
                mu.append(int(j))
            Nb_recharge = int(Nb_recharge)#conversion en entier de Nb_recharge
            for k in range(Nb_recharge):
                F.append(mm_list_int[k])
                F.append(MM_list_int[k])
                L.append(MM_list_int[k] - mm_list_int[k] + 1)
# à enlever debut
#       j=0
#        mu_act=-1
#        while( j <len(QteRecharg_one_int)):
#            #print(QteRecharg_one_int[j])
#            if(QteRecharg_one_int[j]!=0):
#                mu_act=QteRecharg_one_int[j]
#                mu.append(QteRecharg_one_int[j])
#                deb=j
#                while(QteRecharg_one_int[j]==mu_act):
#                    j=j+1
#                fin=j-1
#                F.append(deb)
#                F.append(fin)
#                L.append(fin-deb+1)
#            else:
#                j=j+1
      #  mu = list(set(mu)) # Convertir liste en set puis cette dernier en
      #  liste
# à enlever fin
        #Nb_recharge = nombre de recharge
            for k in range(Nb_recharge):
                mu_.append(mu[k] / L[k])
            E0_list = int(E0_list.rstrip('\n'))
            lambda_train.append(-1 * E0_list)
            for cpt in range(1,N_actuel):
                kk = 0
                somme_mu_ = 0
                for k in range(0,Nb_recharge):
                    if(cpt >= F[kk] and cpt <= F[kk + 1]):
                        somme_mu_ = somme_mu_ + mu_[k]
                    kk = kk + 2
                lambda_train.append(somme_mu_)
            lambda_train.append(E0_list)
        #if(N<len(lambda_train)):
        #    N=len(lambda_train)
            tmp = np.arange(N_actuel) #[0 1 2 ...  N-1]
            for cpt in range(N_actuel):
                arange_train.append(tmp[cpt])
                #print(type(tmp[cpt]))
            #print(type(arange_train))
        #print(arange_train)
            CV_train.append(Cv_train)
            CF_train.append(Cf_train)
            REnd_train.append(Rend_train)
            Lambda_train.append(lambda_train)
        #print(Lambda_train)
            Arange_train.append(arange_train)
            Y_train.append(y_train)
            Y_LIST_int.append(y_list_int)
            count_Nb_recharge.append(Nb_recharge)
            Date_last_recharg.append(DateDebLastRecharg_one)
            STATS_nb_station.append(int(M_list.rstrip('\n')))#sert à faire les statistiques
            STATS_N.append(int(N_list.rstrip('\n')))#sert à faire les statistiques
            STATS_CF.append(int(cout_fix_list.rstrip('\n')))#sert à faire les statistiques
            STATS_vmax.append(int(vmax_list.rstrip('\n')))#sert à faire les statistiques
            STATS_ctank.append(int(Ctank_list.rstrip('\n')))#sert à faire les statistiques
            STATS_cout_var_moyen.append(np.mean(Cv_train))
            STATS_N_Vr.append(int(N_list.rstrip('\n'))/p_list)
            
        else:
            #print(i)
            kkk = math.ceil(int(N_list.rstrip('\n')) // int(p_list.rstrip('\n')) / N_limite)
            print("Compactation",i)
            rend_one, CoutFixe_one, CoutVar_one, CoutProd_one, DateDebLastRecharg_one, QteRecharg_one = Read_Line_By_Line(chemin_prodRN, i)
            sum = 0
            cpt_nb_list = 0
            for j in CoutVar_one.split():
                sum+=int(j)
                cpt_nb_list+=1
                if(cpt_nb_list == kkk):
                    Cv_train.append(math.ceil(sum / kkk))
                    sum = 0
                    cpt_nb_list = 0
            sum = 0
            cpt_nb_list = 0
            for j in CoutFixe_one.split():
                sum = int(j)
                cpt_nb_list+=1
                if(cpt_nb_list == kkk):
                    Cf_train.append(math.ceil(sum / kkk))
                    sum = 0
                    cpt_nb_list = 0
            sum = 0
            cpt_nb_list = 0
            for j in rend_one.split():
                sum+=int(j)
                cpt_nb_list+=1
                if(cpt_nb_list == kkk):
                    Rend_train.append(math.ceil(sum / kkk))
                    sum = 0
                    cpt_nb_list = 0
            alpha_list = int(alpha_list.rstrip('\n'))
            p_list = int(p_list.rstrip('\n'))
            DateDebLastRecharg_one = int(DateDebLastRecharg_one.rstrip('\n'))
            DateDebLastRecharg_one = math.ceil((DateDebLastRecharg_one // p_list) / kkk)
            CoutProd_one = int(CoutProd_one.rstrip('\n'))
            #Si le dernier élément de la fonction Input_Output_CustomLoss vaut
            #0 alors y_train=last_refuel+coutprod
            #Si le dernier élément de la fonction Input_Output_CustomLoss vaut
            #1 alors y_train=last_refuel
            #Si le dernier élément de la fonction Input_Output_CustomLoss vaut
            #2 alors y_train=coutprod
            if(nature_ytrain == 0):
                y_train = ((alpha_list * (DateDebLastRecharg_one)/kkk) + math.ceil(( CoutProd_one) / kkk))
            if(nature_ytrain == 1):
                y_train = (alpha_list * math.ceil((DateDebLastRecharg_one) / kkk))
            if(nature_ytrain == 2):
                y_train = math.ceil((CoutProd_one) / kkk)
            N_actuel = len(Rend_train)##int(N_list.rstrip('\n'))//p_list14/03/2022
            if(N < N_actuel):
                N = N_actuel
            QteRecharg_one = QteRecharg_one.rstrip('\n')
            #print(QteRecharg_one)
            mu = []
            mu_ = []#mu*
            F = []
            L = []
            deb = -1
            fin = -1
            for j in QteRecharg_one.split():
                QteRecharg_one_int.append(int(j))
            ch_num_file = "%d" % i # Transforme num_file en chaine de caractere
            cheminComplets = chemin_prodRN_suite + ch_num_file + ".txt"
            with open(cheminComplets, "r") as fichier_RS:
                y_list = fichier_RS.readline() #vecteur production qui correspond à y dans le modèle réseau de neurones
                Nb_recharge = fichier_RS.readline() #le nombre de recharge calculé par la partie véhicule du pipeline
                mm_list = fichier_RS.readline() #periode au plutot possible de chaque s ieme recharge
                MM_list = fichier_RS.readline() #periode au plustard possible de chaque s ieme recharge
                muS_list = fichier_RS.readline()#Quantite d'hydrogene recharger par le vehicule a chaque s ieme recharge
            for j in y_list.split():
                y_list_int.append(int(j)) #conversion en entier du vecteur production
            for j in mm_list.split():
                mm_list_int.append(math.ceil(int(j) / kkk))
            for j in MM_list.split():
                MM_list_int.append(math.ceil(int(j) / kkk))
            for j in muS_list.split():#remplisage de la liste mu
                mu.append(math.ceil(int(j) / kkk))
            Nb_recharge = int(Nb_recharge)#conversion en entier de Nb_recharge
            for k in range(Nb_recharge):
                F.append(mm_list_int[k])
                F.append(MM_list_int[k])
                L.append(MM_list_int[k] - mm_list_int[k] + 1)
# à enlever debut
#       j=0
#        mu_act=-1
#        while( j <len(QteRecharg_one_int)):
#            #print(QteRecharg_one_int[j])
#            if(QteRecharg_one_int[j]!=0):
#                mu_act=QteRecharg_one_int[j]
#                mu.append(QteRecharg_one_int[j])
#                deb=j
#                while(QteRecharg_one_int[j]==mu_act):
#                    j=j+1
#                fin=j-1
#                F.append(deb)
#                F.append(fin)
#                L.append(fin-deb+1)
#            else:
#                j=j+1
      #  mu = list(set(mu)) # Convertir liste en set puis cette dernier en
      #  liste
# à enlever fin
        #Nb_recharge = nombre de recharge
            for k in range(Nb_recharge):
                mu_.append(mu[k] / L[k])
            E0_list = int(E0_list.rstrip('\n'))
            lambda_train.append(-1 * E0_list)
            for cpt in range(1,N_actuel):
                kk = 0
                somme_mu_ = 0
                for k in range(0,Nb_recharge):
                    if(cpt >= F[kk] and cpt <= F[kk + 1]):
                        somme_mu_ = somme_mu_ + mu_[k]
                    kk = kk + 2
                lambda_train.append(somme_mu_)
            lambda_train.append(E0_list)
        #if(N<len(lambda_train)):
        #    N=len(lambda_train)
            tmp = np.arange(N_actuel) #[0 1 2 ...  N-1]
            for cpt in range(N_actuel):
                arange_train.append(tmp[cpt])
        #print(arange_train)
            CV_train.append(Cv_train)
            CF_train.append(Cf_train)
            REnd_train.append(Rend_train)
            Lambda_train.append(lambda_train)
        #print(Lambda_train)
            Arange_train.append(arange_train)
            Y_train.append(y_train)
            Y_LIST_int.append(y_list_int)
            count_Nb_recharge.append(Nb_recharge)
            Date_last_recharg.append(DateDebLastRecharg_one)
            STATS_nb_station.append(int(M_list.rstrip('\n')))#sert à faire les statistiques
            STATS_N.append(int(N_list.rstrip('\n')))#sert à faire les statistiques
            STATS_CF.append(int(cout_fix_list.rstrip('\n')))#sert à faire les statistiques
            STATS_vmax.append(int(vmax_list.rstrip('\n')))#sert à faire les statistiques
            STATS_ctank.append(int(Ctank_list.rstrip('\n')))#sert à faire les statistiques
            STATS_cout_var_moyen.append(np.mean(Cv_train))
            STATS_N_Vr.append(int(N_list.rstrip('\n'))/p_list)
            
    temp_Ytrain = Y_train
    temp_Ytrain2=Y_train
    temp_Ytrain3=Y_train
    temp_Ytrain4=Y_train
    temp_Ytrain5=Y_train
    temp_Ytrain6=Y_train
    temp_Ytrain7=Y_train
    temp_Ytrain8=Y_train
    #print(len(Date_last_recharg))
    #print(len(CV_train))
    

    Nb_valtest = len(temp_Ytrain4)
    tailll=len(STATS_CF)
    temp_Ytrain4, STATS_CF =zip(*sorted(zip(temp_Ytrain4, STATS_CF)))
    #plt.plot(x, temp_Ytrain4, "g", label="Coût de production optimal")
    STATS_CF_by_100=[]#STATS_CF_by_100 est la liste obtenues après transformation de la liste STATS_CF en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_CF_by_100.append(np.mean(STATS_CF[100*k:100*k+100]))
    STATS_CF_by_100.append(np.mean(STATS_CF[5900:5999]))
    #print(len(STATS_CF_by_100))
    plt.plot(x, STATS_CF_by_100, "b")#, label="Coût fixe de production")
    plt.ylabel('Coût fixe de production')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_coutfix.pdf')
    plt.show()

    Nb_valtest = len(temp_Ytrain8)
    temp_Ytrain8, STATS_N_Vr =zip(*sorted(zip(temp_Ytrain8, STATS_N_Vr)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain8, "g", label="Coût de production optimal")
    STATS_N_Vr_by_100=[]#STATS_N_Vr_by_100 est la liste obtenues après transformation de la liste STATS_N_Vr en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_N_Vr_by_100.append(np.mean(STATS_N_Vr[100*k:100*k+100]))
    STATS_N_Vr_by_100.append(np.mean(STATS_N_Vr[5900:5999]))
    #print(len(STATS_N_Vr_by_100))
    plt.plot(x, STATS_N_Vr_by_100, "b")#, label="Nombre de périodes")
    plt.ylabel('Nombre de périodes')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_nb_periods.pdf')
    plt.show()

    Nb_valtest = len(temp_Ytrain)
    temp_Ytrain, count_Nb_recharge = zip(*sorted(zip(temp_Ytrain, count_Nb_recharge)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain, "g", label="Coût de production optimal")
    count_Nb_recharge_by_100=[]#count_Nb_recharge_by_100 est la liste obtenues après transformation de la liste count_Nb_recharge en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        count_Nb_recharge_by_100.append(np.mean(count_Nb_recharge[100*k:100*k+100]))
    count_Nb_recharge_by_100.append(np.mean(count_Nb_recharge[5900:5999]))
    #print(len(count_Nb_recharge_by_100))
    plt.plot(x, count_Nb_recharge_by_100, "b")#, label="Nombre de recharge")
    plt.ylabel('Nombre de recharge')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_Nbrecharge.pdf')
    plt.show()

    Nb_valtest = len(temp_Ytrain3)
    temp_Ytrain3, STATS_nb_station = zip(*sorted(zip(temp_Ytrain3, STATS_nb_station)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain3, "g", label="Coût de production optimal")
    STATS_nb_station_by_100=[]#STATS_nb_station_by_100 est la liste obtenues après transformation de la liste STATS_nb_station en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_nb_station_by_100.append(np.mean(STATS_nb_station[100*k:100*k+100]))
    STATS_nb_station_by_100.append(np.mean(STATS_nb_station[5900:5999]))
    #print(len(STATS_nb_station_by_100))
    plt.plot(x, STATS_nb_station_by_100, "b")#, label="Nombre de stations")
    plt.ylabel('Nombre de stations')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_Nbstations.pdf')
    plt.show()

    Nb_valtest = len(temp_Ytrain2)
    temp_Ytrain2, STATS_N = zip(*sorted(zip(temp_Ytrain2, STATS_N)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain2, "g", label="Coût de production optimal")
    STATS_N_by_100=[]#STATS_N_by_100 est la liste obtenues après transformation de la liste STATS_N en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_N_by_100.append(np.mean(STATS_N[100*k:100*k+100]))
    STATS_N_by_100.append(np.mean(STATS_N[5900:5999]))
    #print(len(STATS_N_by_100))
    plt.plot(x, STATS_N_by_100, "b")#, label="TMax")
    plt.ylabel('TMax')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_Tmax.pdf')
    plt.show()
    
    Nb_valtest = len(temp_Ytrain5)
    temp_Ytrain5, STATS_vmax = zip(*sorted(zip(temp_Ytrain5, STATS_vmax)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain5, "g", label="Coût de production optimal")
    STATS_vmax_by_100=[]#STATS_vmax_by_100 est la liste obtenues après transformation de la liste STATS_vmax en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_vmax_by_100.append(np.mean(STATS_vmax[100*k:100*k+100]))
    STATS_vmax_by_100.append(np.mean(STATS_vmax[5900:5999]))
    #print(len(STATS_vmax_by_100))
    plt.plot(x, STATS_vmax_by_100, "b")#, label="Capacité du reservoir")
    plt.ylabel('Capacité du reservoir')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_reservoir.pdf')
    plt.show()

      
    Nb_valtest = len(temp_Ytrain6)
    temp_Ytrain6, STATS_ctank =zip(*sorted(zip(temp_Ytrain6, STATS_ctank)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain6, "g", label="Coût de production optimal")
    STATS_ctank_by_100=[]#STATS_ctank_by_100 est la liste obtenues après transformation de la liste STATS_ctank en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_ctank_by_100.append(np.mean(STATS_ctank[100*k:100*k+100]))
    STATS_ctank_by_100.append(np.mean(STATS_ctank[5900:5999]))
    #print(len(STATS_ctank_by_100))
    plt.plot(x, STATS_ctank_by_100, "b")#, label="Capacité de la citerne")
    plt.ylabel('Capacité de la citerne')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_citerne.pdf')
    plt.show()

    Nb_valtest = len(temp_Ytrain7)
    temp_Ytrain7, STATS_cout_var_moyen =zip(*sorted(zip(temp_Ytrain7, STATS_cout_var_moyen)))
    #x = np.linspace(1,Nb_valtest,Nb_valtest) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    #plt.plot(x, temp_Ytrain7, "g", label="Coût de production optimal")
    STATS_cout_var_moyen_by_100=[]#STATS_cout_var_moyen_by_100 est la liste obtenues après transformation de la liste STATS_cout_var_moyen en la regroupant par paquet de 100 
    Nb_valtest_by_100=Nb_valtest//100
    x = np.linspace(1,Nb_valtest_by_100,Nb_valtest_by_100) #on a N valeurs sur l'axe des x (une pour chaque vecteur)
    for k in range(Nb_valtest_by_100-1):
        STATS_cout_var_moyen_by_100.append(np.mean(STATS_cout_var_moyen[100*k:100*k+100]))
    STATS_cout_var_moyen_by_100.append(np.mean(STATS_cout_var_moyen[5900:5999]))
    #print(len(STATS_cout_var_moyen_by_100))
    plt.plot(x, STATS_cout_var_moyen_by_100, "b")#, label="Coût de producion moyen")
    plt.ylabel('Coût de producion moyen')
    plt.xlabel('numéro d\'instances triées par coût de production croissant')
    #plt.legend()
    plt.savefig('Stats_instances_cout_var.pdf')
    plt.show()
   
    

    return CV_train, CF_train, REnd_train, Lambda_train, Arange_train, Y_train, N, Y_LIST_int, count_Nb_recharge, Date_last_recharg,kkk
    
def Transforme_list__same_tail(Cv_train,N):
    #on transforme chaque liste en liste de taille N en completant la liste pas des N-len(Cv_train[i]) de la liste
    #exemple : 1 2 3 4 1 2 3
    for i in range(0, len(Cv_train)):
        #print(len(Cv_train[i]))
        list_0=[]#liste qui complètera la liste initiale
        if(len(Cv_train[i]) < N):
            j=0
            k=0
            while j<N-len(Cv_train[i]):
                list_0.append(Cv_train[i][k])
                j=j+1
                k=k+1
                if(k==len(Cv_train[i])):
                   k=0
            Cv_train[i].extend(list_0)
        if(len(Cv_train[i])>N):
            print("!!!!!!!!!!!!!!Transforme_list__same_tail : cette liste n'a pas la même longueur que les autres")
            
            #print(len(Cv_train[i]))
            #print(Cv_train[i])
    return Cv_train
