import tensorflow as tf
from numpy import column_stack
import matplotlib.pyplot as mt
from matplotlib import style  

#Function for correlation (i.e., <O[g]O[i]>, where i=1,...,L and g!=g)
def correlation(O,L,op,g):


    #setting initial condition 
    op2=1
    p=0
    t=[]
    
    # Reshaping 1st and last site mps for calculation of correlation demanded by the code
    op[0]=tf.reshape(op[0],([1,len(op[0]),len(op[0][0])]))
    op[L-1]=tf.reshape(op[L-1],([len(op[L-1]),len(op[L-1][0]),1]))


    # for loop to calculate correlation function with respect to first site MPS
    for i in range(0,L-1,1):


        # for loop to calculate correlation function of operator 'O' between ith and (i+m)th site MPS
        for m in range(0,L-i,1):

            #setting initial conditions
            opl=tf.constant([[1]],dtype=tf.double)
            opr=tf.constant([[1]],dtype=tf.double)


            #for loop to take innerr product of MPS starting from left hand side till ith site
            for j in range(0,i):
                a=tf.einsum('ik,ikmn->mn',opl,tf.einsum('ijm,kjn->ikmn', op[j], op[j]))
                opl=a
                
            
            #for loop to take innerr product of MPS starting from right hand side till mth site
            for k in range(0,L-i-m-1):
                c=tf.einsum('ikmn,mn->ik',tf.einsum('ijm,kjn->ikmn', op[L-k-1], op[L-k-1]),opr)
                opr=c
            

            #setting if condition such that negative m value need to be ignored
            if m!=0:
    

                # operating 'O' on (i+m)th site of MPS
                op3=tf.einsum('mikl,kl->mi',tf.einsum('mlk,ilj->mikj' ,tf.einsum('mjk,jl->mlk', op[i+m], O),op[i+m]),opr)
                

                # for loop to take inner product of MPS lie between ith and (i+m)th site 
                for l in range(0,m-1):
                    w= tf.einsum('ikmn,mn->ik',tf.einsum('ijm,kjn->ikmn', op[i+m-1-l], op[i+m-1-l]),op3)
                    op3=w


            else:

                # negative m loops are ignored
                # operating 'O' on (i+m)th site of MPS
                op3=tf.einsum('mikl,kl->mi',tf.einsum('mlk,ilj->mikj' ,tf.einsum('mjk,jl->mlk', op[i+m], O),tf.einsum('jl,mjk->mlk',O,op[i+m])),opr)
                

                # for loop to take inner product of MPS lie between ith and (i+m)th site 
                for l in range(0,m-1):
                    w= tf.einsum('ikmn,mn->ik',tf.einsum('ijm,kjn->ikmn', op[i+m-1-l], op[i+m-1-l]),op3)
                    op3=w


            #operator operating on specific site
            if m!=0:
                op2= tf.einsum('mn,mlk->nlk' ,opl,tf.einsum('mjk,jl->mlk', op[i], O))                               
                op2=tf.einsum('nlk,kp->nlp' ,op2,op3)


            # correlation of operator 'O' between ith and (i+m)th site 
            if m!=0:
                b = tf.tensordot(op2,op[i],axes=([0,1,2],[0,1,2]))

            else:
                b = tf.tensordot(opl,op3,axes=([0,1],[0,1]))



            # Collecting correlation data with respect to lth site and other site
            if ((i<g-1) and (m==g-1-i)):
                t.append(b)
                p+=1

            if i==g-1: 
                t.append(b)
                p+=1


    #plotting of correlation function     
    style.use('ggplot')
    o=[i for i in range(0,p)]
    u=[l for i in range(1,p)]

    data = column_stack([o,t])
    #np.savetxt("co11Jz-115o.out", data, fmt=['%lf','%lf'])             # Saving correlation and two sites of 
                                                                            # operation data in the .out file
    
    mt.plot(o,t)
    mt.ylabel("Correlation (<Sz(1)Sz(l)>)",fontsize = 18)
    mt.xlabel("l$^{th}$ site",fontsize = 18)
    mt.show()        
    
    return t
