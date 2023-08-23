from tensorflow import einsum
from tensorflow import reshape
from numpy import shape

# contracting each site MPOs with same site mps except for lth-site mps for getting 
# matrice 'O' from eigen value equestion (i.e., OX-Lambda*NX=0) from text.
def mixEV(O,mps,L,l):

    # L is the size of the chain
    # O is the set of MPOs
    # mps is the set of mps

    # Solving for L matrice from the text
    if l!=1:

        mps[0]=einsum('jlm,jn->nlm' ,einsum('jnl,jm->nlm', O[0], mps[0]), mps[0])

        for i in range(1,l-1,1):
            mps[i]=einsum('kinlm,onp->okiplm' ,einsum('kjnl,ijm->kinlm', O[i], mps[i]), mps[i])

        Left=mps[0]

        for i in range(1,l-1,1):
            
            Left=einsum('lmn,lmnqpr->qpr',Left,mps[i])              # L matrice
    

    # Solving for R matrice from the text
    if l!=L:

        mps[L-1]=einsum('mjn,pn->mjp' ,einsum('jnl,ml->mjn', O[L-1], mps[L-1]), mps[L-1])
        
        for i in range(L-2,l-1,-1):
            mps[i]=einsum('iknml,onp->ikomlp' ,einsum('kjnl,inm->ikjml', O[i], mps[i]), mps[i])
        
        Right=mps[L-1]

        for i in range(L-2,l-1,-1):
            Right=einsum('lmnrpq,rpq->lmn',mps[i],Right)            # R matrice
    

    # Solving for Square matrice 'O' from the text
    ev1=1


    # ev = Matrice 'O' & ev1 would be used for calculation of energy 
    # expectation value for many body quantum operator
    if l!=1 and l==L: 

        ev=einsum('ijn,jkl->ilnk',Left, O[l-1])

        ev=reshape(ev,(shape(ev)[0]*shape(ev)[1], shape(ev)[2]*shape(ev)[3]))
    
    elif l!=L and l==1:

        ev1=einsum('kjl,ilm->jikm',O[l-1],Right)

        ev=reshape(ev1,(shape(ev1)[0]*shape(ev1)[1],shape(ev1)[2]*shape(ev1)[3]))
        
    else:

        ev=einsum('nkilm,rmt->nkrilt',einsum('nji,jlkm->nkilm', Left, O[l-1]),Right)
        
        ev=reshape(ev,(shape(ev)[0]*shape(ev)[1]*shape(ev)[2],shape(ev)[3]*shape(ev)[4]*shape(ev)[5]))
    

    return ev,ev1
        