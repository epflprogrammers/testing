import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import svd
import scipy
    
#defining the operating functions


#S+S- operator
def splsmi(vec,i):
    vec[i]=vec[i]+1
    vec[i+1]=vec[i+1]-1
    return vec
    
#S-S+ operator
def smispl(vec,i):
    vec[i]=vec[i]-1
    vec[i+1]=vec[i+1]+1
    return vec

#SzSz operator

def szsz(vec,i):
    if vec[i]==vec[i+1]:
        return True
    else:
        return False
    
#A function to get vector space for the lattice of desired length

def vec_lat(L):
    
    # Generate spin half vectors of length N

    N=2**L

    #Creating a single vector

    v=[1/2 for i in range(L)]

    #vector space

    vs=[]


    #Creating the real list of all the vectors

    for i in range(N):
        binary=bin(i)
        a=v.copy()
        for k in range(len(str(binary))):
            if binary[-k-1]=="1":
                a[-k-1]=-0.5
    
        vs.append(a)      #vs is the vector space
    
    return vs

    
#A function to generate a hamilitonian for a lattice of length L

def ham_lat(L):

    # Generate spin half vectors of length N

    N=2**L

    #Creating a single vector

    v=[1/2 for i in range(L)]

    #vector space

    vs=[]


    #Creating the real list of all the vectors

    for i in range(N):
        binary=bin(i)
        a=v.copy()
        for k in range(len(str(binary))):
            if binary[-k-1]=="1":
                a[-k-1]=-0.5
    
        vs.append(a)      #vs is the vector space
   


    


#defining the hamiltonian operator

    H=np.zeros((N,N))

#Filling the hamiltonian operator

    for l in range(N):
        for m in range(N):
        #S+S-
            for i in range(L-1):
                dummy=vs[m].copy()
                vec3=splsmi(dummy,i)
                if vs[l]==vec3:
                    H[l][m]=H[l][m]+0.5
        #S-S+
        
            for i in range(L-1):
                dummy=vs[m].copy()
                vec3=smispl(dummy,i)
                if vs[l]==vec3:
                    H[l][m]=H[l][m]+0.5
                
        #SzSz
            for i in range(L-1):
                dummy=vs[m].copy()
                vec3=dummy
                if vs[l]==vec3:
                    if szsz(vec3,i)==True:
                        H[l][m]=H[l][m]+0.25
                    else:
                        H[l][m]=H[l][m]-0.25
                
        
                
                
    return H

#GS Energy vs length plot 



def en_len(S,E):
    list1=[]
    list2=[]
    for L in range(S,E):
      H=ham_lat(L)
      eig,evals=la.eig(H)
      eig1=np.sort(eig)
      list1.append(L)
      list2.append(eig1[0])
      plt.xlabel("Number of sites")
      plt.ylabel("GS Energy")
      plt.title("GS energy vs no. of sites for a spin 1/2 heisenberg chain")
      plt.plot(list1,list2)
        

        






#To get the n'th highest normalised eigenvector

def inc_evecs(H,level):
    eig,evecs=la.eig(H)
    eig1=eig.copy()
    eig1.sort()
    eig=eig.tolist()
    ind=eig.index(eig1[level])
    #required eigenvector
    return eig1[level],evecs[:,ind]


#Performing singular value decomposition of a matrix and getting U, D and V.

def SVD(psi):
    X1=np.dot(psi,psi.T)
    X2=np.dot(psi.T,psi)
    eig1,evecs1=la.eig(X1)
    sorts1=eig1.argsort()[::-1]
    eig2,evecs2=la.eig(X2)
    sorts2=eig2.argsort()[::-1]
    
    
    
    #calculating U
    N=len(psi)
    U=np.zeros((N,N))
    
    for i in range(len(psi)):
        for j in range(len(psi)):
            U[j][i]=evecs1[j][sorts1[i]]
            
    
    
    #calculating V
    N1=len(psi[0])
    V=np.zeros((N1,N1))
    
    for i in range(len(psi[0])):
        for j in range(len(psi[0])):
            V[j][i]=evecs2[j][sorts2[i]]

   
    
    
    #calculating D
    D=np.zeros((N,N1))
    eig1=np.sort(eig1)[::-1]
    for i in range(min(N,N1)):
        D[i][i]=np.sqrt(eig1[i])
            
          
    
    return U,D,V
    
    

#To calculate the entropy of the system

def entropy(H,level,sys_A,sys_B):
    eig,evec=inc_evecs(H,level)
    psi=evec.reshape(2**sys_A,2**sys_B)
    
    
    U,D,V=SVD(psi)

    entro=0
    for i in range(len(D)):
        if D[i][i]==0:
            pass
        else:
            entro=entro-D[i][i]**2*np.log(D[i][i]**2)
    return entro



#To calculate the entropy given PSI

def entropy_psi(psi,sys_A,sys_B):
    si=psi.reshape(2**sys_A,2**sys_B)
    U,D,V=SVD(si)

    entro=0
    for i in range(len(D)):
        if D[i][i]==0:
            pass
        else:
            entro=entro-D[i][i]**2*np.log(D[i][i]**2)
    return entro
 
#Function to take the ground state and decompose it into matrix product states

def MPS(psi):
    L=len(psi)
    L=int(np.log2(L))
    
    prod_sts=[]
    for i in range(L-1):
        evec=psi.reshape(2,2*(L-1-i))
        U,D,V=SVD(evec)
        prod_sts.append(U[:,0])
        psi=V[:,0]
        
    prod_sts.append(V[:,0])
    return prod_sts

#Performing corrected singular value decomposition of a matrix and getting U, D and V.
'''
def cSVD(psi):
    
    X2=np.dot(psi.T,psi)
    

    eig2,evecs2=la.eig(X2)
    sorts2=eig2.argsort()[::-1]
    
    

    
    N=len(psi)
    
    
    #calculating V
    N1=len(psi[0])
    V=np.zeros((N1,N1))
    
    for i in range(len(psi[0])):
        for j in range(len(psi[0])):
            V[j][i]=evecs2[j][sorts2[i]]

   
    
    
    #calculating D
    D=np.zeros((N,N1))
    eig2=np.sort(eig2)[::-1]
    for i in range(min(N,N1)):
        D[i][i]=np.sqrt(eig2[i])
    
    #calculating U
    
    N=len(psi)
    U=np.zeros((N,N))
    
    for i in range(min(N,N1)):
        if np.round(D[i,i],4)!=0:
            U[:,i]=np.dot(psi,V[:,i])/D[i,i]
        else:
            pass
    
            
          
    
    return U,D,V

'''
def S_dia(U,S,Vt):
    sigma=np.zeros((len(U),len(Vt)))
    for i in range(min(len(U),len(Vt))):
        sigma[i][i]=S[i]
    return sigma



def cSVD(psi):
        try:
            U,S,Vt=scipy.linalg.svd(psi,lapack_driver='gesdd')
            return U,S_dia(U,S,Vt),Vt.T.conj()
        except:
            print("gesvd")
            U,S,Vt=scipy.linalg.svd(psi,lapack_driver='gesvd')
            return U,S_dia(U,S,Vt),Vt.T.conj()
	



