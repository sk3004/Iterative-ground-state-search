import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mt
from matplotlib import style

#Function for local expectaion value (i.e., <O[i]>)
def operator(O,L,op):
    
    # O is the local operator whose local expectaion value is asked to be calculated
    # L is the total size of the chain
    # op are matrix product states

    #setting initial conditions
    al1=0
    t=[]
    op2=1
    p=1

    #For looping for calculation local expection value
    for i in range(0,L,1):
        
        #for loop to take innerr product of MPS starting from left hand side till the site where local expectation value is suppose to e measure
        if i!=0:    
            opl=np.einsum('jm,jn->mn', op[0], op[0])
        for j in range(1,i):
            opl=np.einsum('ik,ikmn->mn',opl,np.einsum('ijm,kjn->ikmn', op[j], op[j]))

        #for loop to take innerr product of MPS starting from right hand side till the site where local expectation value is suppose to e measure    
        if i!=L-1:
            opr=np.einsum('ij,kj->ik', op[L-1], op[L-1])
        for k in range(1,L-i-1):
            opr=np.einsum('ikmn,mn->ik',np.einsum('ijm,kjn->ikmn', op[L-k-1], op[L-k-1]),opr)
            

        #operator operating on specific sites   
        if i!=0 and i!=L-1: 
            op2= np.einsum('mjk,lj->mlk', op[i], O)                                                     
            op2= np.einsum('mn,mlk->nlk',opl,np.einsum('mli,ik->mlk',op2,opr))
            b = tf.tensordot(op2,op[i],axes=([0,1,2],[0,1,2]))
        

        if i==0:
            op2= np.einsum('jk,lj->lk', op[i], O)                                                     
            op2= np.einsum('li,ik->lk',op2,opr)
            b = tf.tensordot(op2,op[i],axes=([0,1],[0,1]))
            

        if i==L-1: 
            op2= np.einsum('mj,lj->ml', op[i], O)                                                     
            op2= np.einsum('mn,nl->ml',opl,op2)
            b= np.tensordot(op2,op[i],axes=([0,1],[0,1]))


        #Appending all local expectaion value
        t.append(b)
        
        p+=1


        #Adding absolute value of all the local expectaion values coming from each sites MPS
        al1+=abs(b)

    #Plotting local expectaion value
    o=[i for i in range(1,p)]

    # data = np.column_stack([o,t])                                     
    # np.savetxt("levJz-1J60.out", data, fmt=['%lf','%lf'])             # Saving local expectation value and  
                                                                        # corresponding site data in .out file

    style.use('ggplot')
    mt.plot(o,t)
    mt.ylabel("Local Expectation Value (<Sz(l)>)",fontsize = 18)
    mt.xlabel("lth site",fontsize = 18)
    mt.savefig('SV.png')
    mt.show()


    return al1