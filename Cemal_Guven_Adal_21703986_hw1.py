#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt #importing matplot library for plots
import numpy as np # importing numpy library for easier math operations
import math # importing math library for easier computations
import sys


def CemalGuven_Adal_21703986_hw1(question):
    if question == '1' :
        #---------------------------------------------------------------------------Question1--------------------------------------------------------------------
        #Question 1 A------------------------------------------
        #Defining array A which is given in the Homework question 1
        A= np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])
        A
        
        
        
        
        #creating the free variables in order the check the hand solved solution
        x3=np.random.rand()
        x4=np.random.rand()
        #writing general solution xn in terms of free variables x3 and x4
        xn= np.array([[x3-2*x4],[-x3-x4],[x3],[x4]])
        y=np.dot(A,xn)
        print(y)
        
        #Question1 B-------------------------------------------------
        
        
        #Defining array A which is given in the Homework question 1
        A= np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])
        x_p=np.array([[1],[2],[0],[0]]) #Defining particular solution as I have found in handmade calculations
        print(A.dot(x_p))
        
        #Question1 C-------------------------------------------------
        
        
        #Defining array A which is given in the Homework question 1
        A= np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])
        xall=x_p+xn #Defining all solution as adding particular and general solution
        print(A.dot(xall))
        
        
        #Question1 D-------------------------------------------------
        
        A= np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])#creating A
        Atrans=A.transpose()#taking transpose of A
        AAt=np.dot(A,Atrans)#multiplying to matrices
        evalues, evectors =np.linalg.eig(AAt)
        evalues[2]=0
        print("Eigen values:\n",np.round(evalues,3))
        singvalues=np.sqrt(evalues) #finding singular values of A
        print( "Singular values:\n",singvalues)
        U=evectors
        print("U\n",U)
        
        
        AtA=np.dot(Atrans,A)#multiplying to matrices
        evalues, evectors =np.linalg.eig(AtA)
        evalues[2:]=0
        print(np.real(np.round(evalues,3)))
        singvalues=np.sqrt(evalues) #finding singular values of A
        print(np.real(np.round(singvalues,3)))
        V=evectors
        print(np.real(np.round(evectors,3)))
        
        
        
        
        U,Sigma,Vtranspose = np.linalg.svd(A,full_matrices=True)
        print("U:\n",U)
        print("Sigma:\n",Sigma)
        print("Vtranspose:\n",Vtranspose)
        
        
        
        diagonal=np.zeros((3,4))
        diagonal[0,0]=Sigma[0]
        diagonal[1,1]=Sigma[1]
        diagonal[2,2]=Sigma[2]
        print(diagonal)
        
        
        
        aplus=U@diagonal@Vtranspose
        Sigma=np.round(Sigma,1)
        print(np.round(aplus,1))
        
        
        
        D_plus=np.zeros((4,3)) #taking transpose of diagonal 
        
        D_plus[0,0]=1/diagonal[0,0]
        D_plus[1,1]=1/diagonal[1,1]
        print(D_plus)
        
        
        
        
        #check
        D_plus@diagonal
        
        
        
        
        V=np.transpose(Vtranspose)
        Utranspose=np.transpose(U) #Finding Utranspose from U  which have been fond from np.svd
        Apseudo=V@D_plus@Utranspose#Finding A pseudo
        Apseudo
        
        
        
        
        #6.soru f
        b=np.array([[1],[4],[9]])
        x=Apseudo@b#minimum norm
        x
        
        
        
        sparsolution1 = [[1],[2],[0],[0]]
        A@sparsolution1

        
    elif question == '2' :
        
        #-------------------QUESION2--------------------------------------------------------------------------------------------------

        #writing function for computing bernoulli
        def bernoulli(n,i,p):
            combination=(math.factorial(n))/(math.factorial(i)*math.factorial(n-i))
            ber_result=combination*(p**i)*((1-p)**(n-i))
            return ber_result
        
        possible=np.linspace(0,1,1001) #xl and xnl values between 0 and 1 
        like=bernoulli(869,103,possible)#language
        like1=bernoulli(2353,199,possible)#no language
        #Activation plots during language
        
        #plots of likelihood function for involving language 0
        plt.figure()
        
        plt.bar(possible[0:500],like[0:500],width=0.0007)#plot between 0 and 0.5
        plt.title("Likelihood for activation involving language with  probability values between 0 and 0.5")
        plt.xlabel("Probability")
        plt.ylabel("Likelihood")
        #Activation plots during no language
        plt.figure()
        
        plt.bar(possible[0:250],like1[0:250],width=0.0006,color='r')#plot between 0 and 0.25
        plt.title("Likelihood for activation not involving language with  probability values between 0 and 0.25")
        plt.xlabel("Probability")
        plt.ylabel("Likelihood")
            
        
        
        
        
        #b findig MLE
        a=np.argmax(like)
        b=np.argmax(like1)
        print("MLE of activation with language",possible[a])
        print("MLE of activation with no language",possible[b])
        
        
        
        
        #c
        #finding sum of likelihood
        suml1=0
        suml2=0
        cdf=np.zeros(1001)
        cdf1=np.zeros(1001)
        conf=np.zeros(2) 
        conf1=np.zeros(2)
        suml1 = np.sum(like)  
             
         
        suml2=np.sum(like1)   
             
        #finding posteriorpfg 
        
        postpdf=like/suml1
        postpdf1=like1/suml2
        
        #plot for posterior distribution with language 
        
        plt.figure()
        plt.bar(possible[0:250],postpdf[0:250],width=0.0006,color='b')#plot between 0 and 0.25
        plt.title("Posterior Distribution of tasks with language")
        plt.xlabel("Probability")
        plt.ylabel("Posterior")
        
        #posterior plot of no language
        
        plt.figure()
        plt.bar(possible[0:250],postpdf1[0:250],width=0.0006,color='r')#plot between 0 and 0.25
        plt.title("Posterior Distribution of tasks with no language")
        plt.xlabel("Probability")
        plt.ylabel("Posterior")
        for i in range(0, len(like)):    
           cdf[i]=np.sum(postpdf[0:i+1]) 
        for i in range(0, len(like1)):    
           cdf1[i]=np.sum(postpdf1[0:i+1])
        
        #cdf plot language
        
        plt.figure()
        plt.bar(possible,cdf,width=0.0007,color='b')
        plt.title("Cumulative Distribution of tasks with language")
        plt.xlabel("Probability")
        plt.ylabel("CDF")
        
        #cdf plot no language
        
        
        plt.figure()
        plt.bar(possible,cdf,width=0.0007,color='r')
        plt.title("Cumulative Distribution of tasks with no language")
        plt.xlabel("Probability")
        plt.ylabel("CDF")
        print(cdf)
        
        
        #calculating confidence bounds
        #for language
        for i in range(0,len(cdf)):
            if 0.023<cdf[i]<0.026:
                conf[0]=possible[i]
            elif 0.973<cdf[i]<0.975:
                conf[1]=possible[i]
                
        #for no language
        for i in range(0,len(cdf)):
            if 0.018<cdf1[i]<0.030:
                conf1[0]=possible[i]
            elif 0.970<cdf1[i]<0.980:
                conf1[1]=possible[i]
        print("conidence interval for tasks involving lanuage","lower bound:",conf[0],"    upper bound:", conf[1])   
        print("conidence interval for tasks not involving lanuage","lower bound:",conf1[0],"    upper bound:", conf1[1])  
        
        
        
        
        
        
        
        postpdf1tr=np.transpose(postpdf1)
        postpdf1tr=postpdf1tr.reshape(1001,1)
        postpdf=postpdf.reshape(1,1001)
        jointpdf=np.dot(postpdf1tr,postpdf)
        
        plt.figure()
        plt.imshow(jointpdf, origin='lower')
        plt.title("Joint Posterior Distribution")
        plt.xlabel("xl")
        plt.ylabel("xnl")
        plt.colorbar()
        postgreater=0
        post1greater=0
        #P(Xl > Xnl|data)and P(Xl > Xnl|data) finding from matrix
        for i in range(0,len(postpdf1)):
            for k in range(0,len(postpdf1)):
                if i<k:
                    postgreater=postgreater+jointpdf[i,k]
                else:
                    post1greater=post1greater+jointpdf[i,k]
        print(" P(Xl > Xnl|data):",postgreater)
        print("P(Xl < Xnl|data):",post1greater)
        
                   
        
            
        
        
        
        
        #f
        mle1=a
        mle2=b
        plang=0.5
        langact=mle1*plang/((mle1+mle2)*plang)
        print(langact)
            
        
        
question=input("enter question number")
CemalGuven_Adal_21703986_hw1(question)






# In[ ]:




