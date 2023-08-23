
# Singular value decomposition python module for getting 
# mixed-canonical matrix product state

import tensorflow as tf
import numpy as np

def svdmix(C,d,l,L,y):

        #C= coefficient tensor
        #d= dimension of single-site Hilbert space
        #l= last site for SVD operation to get left normalized MPS (so mps[>l] are right-canonical mps)
        #L= size of the chain
        #y= Truncation limit of singular value decomposition


        #Giving initial condition to the for loop
        n=1
        output=[]

        if l!=1:
                #For loop for performing Singular Value Decomposition of left normalised MPS 'A':

                #Singular value decomposition of reshaped many-body quantum state coefficient after reshaping 'R1'
                s1,u1,v1= tf.linalg.svd(tf.reshape(C, [n*d,d**(L-1)]), full_matrices=False, compute_uv=True, name=None)


                #Truncating by removing singular value<0.2 :
                S1=np.diag(np.delete(tf.transpose(s1), 
                        np.where((s1 >= 0) & (s1 <= y))[0], axis=0))                    #new singular value Square matrice 
                
                h1=len(S1)                                                              #size of new singular value


                # Deleting extra column of U matrice that do not sum with new S matrice:
                sizeu1=len(u1[0])                                                       #size of column of u1 matrice

                u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)                              #Deleting columns of u1 which are 
                                                                                        #extra for new singular value


                # Deleting extra columns of V matrices:
                sizev1=len(v1[0])
                v1=np.delete(v1, np.s_[h1:sizev1], axis=1)                              #removing column to match with of S1 
                                                                                        #for matrice multiplication


                output.append(tf.reshape(u1,[d,h1]))                                    #Converting u1 to A(aj,d,aj+1) and
                                                                                        # then Appending  first site left 
                                                                                        # normalized MPS


                R2=tf.tensordot(S1,tf.transpose(v1),axes=([1],[0]))                     # Taking dot product of new S1 
                                                                                        # and new v1 matrices

        #for loop for left normalized MPS
        if l!=L:
                for j in range(2,l+1,1):
                        
                        #Reseting initial condition using last calculation on loop:
                        R1=R2
                        n=h1


                        #Singular value decomposition of reshaped many-body quantum state coefficient after reshaping 'R1'
                        s1,u1,v1= tf.linalg.svd(tf.reshape(R1, [n*d,d**(L-j)]), full_matrices=False, compute_uv=True, name=None)


                        #Truncating by removing singular value<0.2 :
                        S1=np.diag(np.delete(tf.transpose(s1), np.where((s1 >= 0) &
                                 (s1 <= y))[0], axis=0) )                               #new singular value Square matrice 
                        
                        h1=len(S1)                                                      #size of new singular value


                        # Deleting extra column of U matrice that do not sum with new S matrice:
                        sizeu1=len(u1[0])                                               #size of column of u1 matrice

                        u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)                      #Deleting columns of u1 which are 
                                                                                        #extra for new singular value


                        # Deleting extra columns of V matrices:
                        sizev1=len(v1[0])
                        v1=np.delete(v1, np.s_[h1:sizev1], axis=1)                      #removing column to match with of S1 for matrice multiplication
                

                        output.append(tf.reshape(u1,[n,d,h1]))                          #reshaping u1 and then appending 

                        R2=tf.tensordot(S1,tf.transpose(v1),axes=([1],[0]))             # Taking dot product of new S1 and new v1 matrices
                
        if l==L:
                for j in range(2,l,1):
                        
                        #Reseting initial condition using last calculation on loop:
                        R1=R2
                        n=h1


                        #Singular value decomposition of reshaped many-body quantum state coefficient
                        s1,u1,v1= tf.linalg.svd(tf.reshape(R1, [n*d,d**(L-j)]), full_matrices=False,
                                 compute_uv=True, name=None)


                        #Truncating by removing singular value<0.2 :
                        S1=np.diag(np.delete(tf.transpose(s1), np.where((s1 >= 0) 
                                & (s1 <= y))[0], axis=0))                               #new singular value Square matrice 
                        
                        h1=len(S1)                                                      #size of new singular value


                        # Deleting extra column of U matrice that do not sum with new S matrice:
                        sizeu1=len(u1[0])                                               #size of column of u1 matrice

                        u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)                      #Deleting columns of u1 which 
                                                                                        #are extra for new singular value


                        # Deleting extra columns of V matrices:
                        sizev1=len(v1[0])

                        v1=np.delete(v1, np.s_[h1:sizev1], axis=1)                      #removing column to match with of 
                                                                                        #S1 for matrice multiplication
                
                        output.append(tf.reshape(u1,[n,d,h1]))                          #Converting u1 to A(aj,d,aj+1)

                        R2=tf.tensordot(S1,tf.transpose(v1),axes=([1],[0]))             # Taking dot product of new S1 and new v1 matrices

                output.append(tf.reshape(R2,[h1,d]))

        if l!=1:
                h2=h1        

        if l==1:       
                h2=1

                R2=C

        n=1
        if l<L-1:

                #Singular value decomposition of reshaped many-body quantum state coefficient
                s1,u1,v1= tf.linalg.svd(tf.reshape(R2, [h2*d**(L-l),n*d]),
                         full_matrices=False, compute_uv=True, name=None)
                

                #Truncating by removing singular value<y :
                S1=np.diag(np.delete(tf.transpose(s1), 
                        np.where((s1 >= 0) & (s1 <= y))[0], axis=0))            #new singular value Square matrice 
                
                h1=len(S1)
                
                # Deleting extra columns of V matrices:
                sizev1=len(v1[0])                                               #size of column of v1 matrice

                v1=np.delete(v1, np.s_[h1:sizev1], axis=1)                      #removing column to match with of S1 
                                                                                #for matrice multiplication
              
                
                # Deleting extra column of U matrice that do not sum with new S matrice:
                
                sizeu1=len(u1[0])                                               #size of column of u1 matrice

                u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)                      #Deleting columns of u1 which are extra 
                                                                                #for new singular value

                
                output.append(tf.reshape(tf.transpose(v1),[h1,d]))              #Reshaping transpose of v1 and then appending
                
                R2=tf.tensordot(u1,S1,axes=([1],[0]))                           #Taking dot product of new u1 and new S1 matrices
                


        for j in range(1,L-l,1):
                #Reseting initial condition using last calculation on loop:
                R1=R2
                n=h1


                #Singular value decomposition of reshaped many-body quantum state coefficient
                s1,u1,v1= tf.linalg.svd(tf.reshape(R1, [h2*d**(L-l-j),n*d])
                        , full_matrices=False, compute_uv=True, name=None)
                

                #Truncating by removing singular value<0.2 :
                S1=np.diag(np.delete(tf.transpose(s1), 
                        np.where((s1 >= 0) & (s1 <= y))[0], axis=0))            #new singular value Square matrice 
                
                h1=len(S1)
                

                # Deleting extra columns of V matrices:
                sizev1=len(v1[0])

                v1=np.delete(v1, np.s_[h1:sizev1], axis=1)                      #removing column to match with of S1
                                                                                #for matrice multiplication
                

                # Deleting extra column of U matrice that do not sum with new S matrice:
                sizeu1=len(u1[0])                                               #size of column of u1 matrice

                u1=np.delete(u1, np.s_[h1:sizeu1], axis=1)                      #Deleting columns of u1 which are extra
                                                                                #for new singular value
                
                output.append(tf.reshape(tf.transpose(v1),[h1,d,n]))            #Reshaping transpose of v1 and then appending

                R2=tf.tensordot(u1,S1,axes=([1],[0]))                           #Taking dot product of new u1 and new S1 matrices
                
                
        #Solving for final 'B(site= l+1)' Matrice from final R2:
        if l<L-1 and l!=1:
                output.append(tf.reshape(R2,[h2,d,h1])) 

        elif l==L-1:
                output.append(tf.reshape(R2,[h2,d]))

        elif l==1:
                output.append(tf.reshape(R2,[d,h1]))
        
        return output