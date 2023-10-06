import tensorflow as tf
import numpy as np

# Matrix product operator (MPO) for Heisenberg XXZ model
def mpo2(h,J,Jz,L):
    
    # L is the size of the chain

    MPOs=[]                                     # MPO data 

    # Pauli matrices
    Sz=np.array([[1,0],[0,-1]],dtype=None)/2    #Pauli spin-z matrice
    Sp=np.array([[0,0],[1,0]],dtype=None)       #Pauli spin-raising matrice
    Sn=np.array([[0,1],[0,0]],dtype=None)       #Pauli spin-lowering matrice
    Sx=np.array([[0,1],[1,0]],dtype=None)       #Pauli spin-x matrice
    I=np.array([[1,0],[0,1]],dtype=None)        #Identity matrice


    # MPO generation
    W1= np.zeros((2,2,5),dtype=None)            # 1st site MPO
    WL=np.zeros((5,2,2),dtype=None)             # Last site MPO
    Wi=np.zeros((5,2,2,5),dtype=None)           # MPOs between first and last site


    # For loop for setting each elements in MPOs
    for i in range(0,2):
        for j in range(0,2):

            # Setting elements of 1st site MPO
            W1[i][j][0]=-h*Sz[i][j]
            W1[i][j][1]=J*Sn[i][j]/2
            W1[i][j][2]=J*Sp[i][j]/2
            W1[i][j][3]=Jz*Sz[i][j]
            W1[i][j][4]=I[i][j]


            # Setting elements of last site MPO
            WL[0][i][j]=I[i][j]
            WL[1][i][j]=Sp[i][j]
            WL[2][i][j]=Sn[i][j]
            WL[3][i][j]=Sz[i][j]
            WL[4][i][j]=-h*Sz[i][j]


            # Setting elements of MPOs between 1st and last site
            Wi[0][i][j][0]=I[i][j]
            Wi[1][i][j][0]=Sp[i][j]
            Wi[2][i][j][0]=Sn[i][j]
            Wi[3][i][j][0]=Sz[i][j]
            Wi[4][i][j][0]=-h*Sz[i][j]
            Wi[4][i][j][1]=J*Sn[i][j]/2
            Wi[4][i][j][2]=J*Sp[i][j]/2
            Wi[4][i][j][3]=Jz*Sz[i][j]
            Wi[4][i][j][4]=I[i][j]

    

    # Appending MPOs representing each site
    MPOs.append(W1)

    for i in range(0,L-2,1):
        MPOs.append(Wi)

    MPOs.append(WL)

    return MPOs
