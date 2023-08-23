
# Main python module for iterative ground-state method

import tensorflow as tf
import numpy as np
import array as ar
import matplotlib.pyplot as mt
from matplotlib import style

#Importing modules
import MixSVD
import mixEV1
import correlation as co
import Local_expectation_value as lev
import mpo2

# Setting all the necessary parameters

d=2                                                         #Dimension of single-site Hilbert space

L=13                                                         #Total number of site

l=1                                                         #setting last site for SVD operation to get left normalized MPS

y=0.00                                                      # Truncation limit of singular value decomposition
                                                            # for initial mps set calculation

y1=0.00                                                     # Truncation limit of singular value decomposition
                                                            # for iterative ground-state search

g=1                                                         # correlation plot with respect to single site 'g' and 
                                                            # remain site (i.e., <O[g]O[i]>, where i=1,...,L and g!=g)

# Coefficients for Heisenberg XXZ model:
Jz=1.0
J=5.0
h=0.0


#Creating a random tensor of rank 'L':

ten = ar.array('i',(d for i in range(0,L)))                 #creating an array
c = tf.random.uniform(
    (ten),
    minval=-1,
    maxval=1,
    dtype=tf.double,
    seed=None,
    name=None
)


#Normalizing 'c':

C=c/(tf.transpose(tf.reshape(c,(d**L,1)))@                  # 'C' is normalised coefficient tensor of many-body quantum state
   tf.reshape(c,(d**L,1)))**0.5


# performing singular value decomposion for getting Matrix product states (mps):
mps = MixSVD.svdmix(C,d,l,L,y)

mps1=mps*1
for i in range(1,L+1):
    mps[L-i]=mps1[i-1]


# Matrix prodect operator (MPOs) for Heisenberg XXZ model

MPOs=[]
MPOs=mpo2.mpo2(h,J,Jz,L)

# Iterative ground-state search:

gev=[]                                                      # Energy expectation value/L data per iteration


# Energy expectation value/L calculation from initial set of mps:
mps1=mps*1
mps2=mps*1
mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,1)                       # Square matrice acquired by contracting each site MPOs 
                                                            # with same site mps except for the first-site mps

gse=tf.tensordot(mps[0],tf.einsum('jikm,km->ji',
    ev1,mps[0]),axes=([0,1],[0,1]))/L                       # Energy expectation value/L

gev.append(gse)                                             # Noting the first energy expectation value 
                                                            # data before iterative search


# Initiating iterative ground-state search:
h1=0

for i in range(50):

    #setting initial condition:
    gsei=gse

    # Sweeping loop starting from first (leftmost) site to Lth (rightmost) site:
    for j in range(1,L,1):
        h2=h1

        mps1=mps*1

        mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,j)               # Square matrice acquired by contracting each site MPOs 
                                                            # with same site mps except for jth-site mps
        
        ev,evec=np.linalg.eigh(mixEV)                       # Eigen solver solving for lowest eigen value and 
                                                            # correspoding eigen vector for the matrice mixEV


        # Reshaping the solved lowest eigen-value's vector of mixEV for performing SVD in search of new jth site mps:
        if (j==1):
            eisol= tf.reshape(evec[:,0],(np.shape
                (mps[j-1])[0],np.shape(mps[j-1])[1]))

        else:
            eisol= tf.reshape(evec[:,0],(np.shape(
                mps[j-1])[1]*np.shape(mps[j-1])[0],
                np.shape(mps[j-1])[2]))


        # Singular value decomposition of new jth site mps for converting it to Left-canonical mps:
        s1,u1,v1= tf.linalg.svd(eisol, full_matrices=
            False, compute_uv=True, name=None)

        #Truncating by removing singular values < y1 :
        S1=np.diag(np.delete(tf.transpose(s1), 
            np.where((s1 >= 0) & (s1 <= y1))[0], axis=0))       # new singular value Square matrice
       
        h1=len(S1)                                              # size of new singular value


        # Deleting extra column of U matrice that do not sum with new S matrice:
        sizeu1=len(u1[0])                                       # size of column of u1 matrice

        u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)              # Deleting columns of u1 which are 
                                                                # extra for new singular value


        # Deleting extra columns of V matrices:
        sizev1=len(v1[0])                                       # size of column of v1 matrice

        v1=np.delete(v1, np.s_[h1:sizev1], axis=1)              # removing column to match with
                                                                # of S1 for matrice multiplication


        # Reshaping u1 matrice to new left-canonical jth-site mps 
        if j==1:
            #print(i)
            mps[j-1]= tf.reshape(u1,(d,h1))

        elif j!=L:
            mps[j-1]= tf.reshape(u1,(h2,d,h1))


        # Absorbing (j)th-site S1 and v1 matrices to the (j+1)th site mps
        R3=tf.tensordot(S1,tf.transpose(v1),axes=([1],[0]))
        if j!=L-1:
            mps[j]=tf.einsum('li,ijk->ljk',R3,mps[j])


            # Energy expectation value/L calculation 
            # Energy expectation value/L calculation:
            mps1=mps*1
            mps2=mps*1
            mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,1)

            gse=tf.tensordot(mps[0],tf.tensordot(ev1,mps[0],
                axes=([0,1],[0,1])),axes=([0,1],[0,1]))/L
            
            gev.append(gse)                                         # Noting the energy expectation value/L data

    # Absorbing (L-1)th-site S1 and v1 matrices to the Lth site mps
    mps[L-1]=tf.einsum('li,ij->lj',R3,mps[L-1])


    # Energy expectation value/L calculation 
    # Energy expectation value/L calculation:
    mps1=mps*1
    mps2=mps*1
    mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,1)

    gse=tf.tensordot(mps[0],tf.tensordot(ev1,mps[0],
        axes=([0,1],[0,1])),axes=([0,1],[0,1]))/L
    
    gev.append(gse)                                                 # Noting the energy expectation value/L data


    # Sweeping loop starting from Lth (rightmost) site to first (leftmost) site:
    for j in range(L,1,-1):

        h2=h1

        mps1=mps*1
 
        mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,j)                       # Square matrice acquired by contracting each site MPOs 
                                                                    # with same site mps except for jth-site mps
        
        ev,evec=np.linalg.eigh(mixEV)                               # Eigen solver solving for lowest eigen value and 
                                                                    # correspoding eigen vector for the matrice mixEV
        

        # Reshaping the solved lowest eigen-value's vector of mixEV for performing SVD in search of new jth site mps:
        if (j==L):
            eisol= tf.reshape(evec[:,0],(np.shape
                (mps[j-1])[0],np.shape(mps[j-1])[1]))

        elif j!=L:
            eisol= tf.reshape(evec[:,0],(np.shape
                (mps[j-1])[0],np.shape(mps[j-1])[2]
                *np.shape(mps[j-1])[1]))

        # Singular value decomposition of new jth site mps for converting it to Right-canonical mps:
        s1,u1,v1= tf.linalg.svd(eisol, full_matrices
            =False, compute_uv=True, name=None)

        # Truncating by removing singular value < y1:
        S1=np.diag(np.delete(tf.transpose(s1), 
            np.where((s1 >= 0) & (s1 <= y1))[0], axis=0))             #new singular value Square matrice
        
        h1=len(S1)                                                    #size of new singular value

        # Deleting extra column of U matrice that do not sum with new S matrice:
        sizeu1=len(u1[0])                                             #size of column of u1 matrice

        u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)                    #Deleting columns of u1 which are extra for new singular value

        # Deleting extra columns of V matrices:
        sizev1=len(v1[0])
        v1=np.delete(v1, np.s_[h1:sizev1], axis=1)                    #removing column to match with of S1 for matrice multiplication


        # Reshaping v1 matrice to new right-canonical jth-site mps:
        if j==L:
            mps[j-1]= tf.reshape(tf.transpose(v1),(h1,d))

        elif j!=L:
            mps[j-1]= tf.reshape(tf.transpose(v1),(h1,d,h2))


        # Absorbing jth-site S1 and v1 matrices to the (j-1)th site mps:
        R3=tf.tensordot(u1,S1,axes=([1],[0]))

        if j!=2:
            mps[j-2]=tf.einsum('ijk,kl->ijl',mps[j-2],R3)


            # Energy expectation value/L calculation:
            mps1=mps*1
            mps2=mps*1
            mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,1)

            gse=tf.tensordot(mps[0],tf.tensordot(ev1,mps[0],
                axes=([0,1],[0,1])),axes=([0,1],[0,1]))/L
            
            gev.append(gse)                                             # Noting the energy expectation value/L data

    
    # Absorbing 2nd-site S1 and v1 matrices to the 1st site mps:
    mps[0]=tf.einsum('jk,kl->jl',mps[0],R3)


    # Energy expectation value/L calculation:
    mps1=mps*1
    mps2=mps*1
    mixEV,ev1=mixEV1.mixEV(MPOs,mps1,L,1)

    gse=tf.tensordot(mps[0],tf.tensordot(ev1,mps[0],
        axes=([0,1],[0,1])),axes=([0,1],[0,1]))/L
    
    gev.append(gse)                                             # Noting the energy expectation value/L data


    # Setting terminating condition:
    if abs(((gsei-gse)/gsei)*100)<1:
        break


# Ploting energy exectation values/L verses number of iteration
o=[i for i in range(1,len(gev)+1)]

data = np.column_stack([o,gev])
#np.savetxt("gevJz-1J05.out", data, fmt=['%lf','%lf'])          # Saving energy exectation values/L and number 
                                                                # of iteration data in .out file

style.use('ggplot')
mt.plot(o,gev)
mt.ylabel("Energy/L",fontsize = 18)
mt.xlabel("# of iteration",fontsize = 18)
mt.savefig('gse.png')
mt.show()

Sz=tf.constant([[1,0],[0,-1]],dtype=tf.double)                  # Pauli spin-z matrice

# Calculating and plotting local expectation value (LEV) for operator 'Sz' with respect to 
# different sites mps using below functional script:
loev=lev.operator(Sz,L,mps)                                     # Sum of each site's absolute value of the LEV


# Calculating and plotting correlation function for operator 'Sz' with respect to 
# gth site and ith-sites mps (where, i=1,2,3,...,L and i!=g):
cor = co.correlation(Sz,L,mps,g)


print("\n Ground-state energy/L =",gse,"\n\n Sum of each site's absolute value \n of the local expectation value ="
      ,loev,"\n\n Corelation function data:\n\n",tf.reshape(cor,[len(cor),1]))

