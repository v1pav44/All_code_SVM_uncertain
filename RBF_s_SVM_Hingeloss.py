import xdrlib
import numpy as np
import math
from scipy.integrate import quad
import statistics
from scipy import integrate
import random
from sklearn.base import BaseEstimator
import pandas as pd
from numpy.linalg import norm 
import numexpr as ne

class RBF_SVM_Hinge_sgd_2(BaseEstimator):

    def __init__(self, C = 1, max_epochs = 1, n_batches = 2, gamma = 1
                ):
        assert C > 0
        self.C = C
        assert gamma > 0
        self.gamma = gamma
        assert max_epochs > 0 and isinstance(max_epochs, int)
        self.max_epochs=max_epochs
        assert n_batches > 0 and isinstance(n_batches, int)
        self.n_batches=n_batches

    '''gradient of uncertainHingeloss'''

    def sigma(self,x): 
        d = []
        for i in range(len(x[0])):
            m = max(x[:,i]) 
            n = min(x[:,i])
            v = m - n
            d.append(v)

        sig = np.diag(d)
        return sig
        #return sigma

    def kvtest(self,x,t):

        v = []

        for n in range(len(x)):
                
          if t[n] == 1:
                k_0 = random.uniform(0.95*0.05,1.05*0.05)
          elif t[n] == -1:
                k_0 = random.uniform(0.95*0.07,1.05*0.07)

          v.append(k_0)

        return v

    def gradient(self,w, b, x, t, sigma,kv): 
        
        gradient = [] 
        

        for k in range(len(t)):

            dx = 1 - t[k]*(np.dot(w.T,x[k]) + b) 
            sigma_new = 0.25*kv[k]*sigma
            dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
            g1 = (math.exp(-(dx**2)/(dsigma**2))/ (2*math.sqrt(math.pi)*dsigma)) * np.dot(sigma_new,w)      

            f = lambda y : math.exp(-y**2)
            integ = integrate.quad(f, 0, (dx/dsigma))
            g2 = 1/2 * t[k] * ((2/math.sqrt(math.pi))*integ[0] + 1) * x[k]

            g = g1 - g2
            gradient.append(g) 
          
       
        return gradient


    #gradient on b
    def bgradient(self, w, b, x, t, sigma,kv): 
        
        bgradient = [] 
        
              
        for k in range(len(t)):

            dx = 1 - t[k]*(np.dot(w.T,x[k]) + b) 
            sigma_new = 0.25*kv[k]*sigma
            dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
            f = lambda y : math.exp(-y**2)
            integ = integrate.quad(f, 0, (dx/dsigma))
            bg =  1/2 * t[k] * ((2/math.sqrt(math.pi))*integ[0] + 1)

            bgradient.append(bg)

        return bgradient

    def construct_x_uncertain(self,x,t):
        x_new = []

        kv = self.kvtest(x,t)
        #print('kv=', kv)
        sigma = self.sigma(x)
        
        for i in range(len(x)):
            
            sigma_new = 0.25*kv[i]*sigma
  
            cov = sigma_new 
            mean = x[i]

            x_uncertain = np.random.multivariate_normal(mean, cov)
            x_new.append(x_uncertain)
        
        return np.array(x_new)   

     
    def kernel(self,X,Y):
        X_norm = np.sum(X ** 2, axis = -1)
        Y_norm = np.sum(Y ** 2, axis = -1)
        return ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(X, Y.T),
                'g' : self.gamma,
        })

    def cost_function(self,w,b,x,t):
      
      loss = []
      
      kv = self.kvtest(x,t)
      sigma = self.sigma(x)

      for k in range(len(t)):
        dx = 1 - t[k]*(np.dot(w.T,x[k]) + b) 
        sigma_new = 0.25*kv[k]*sigma
        dsigma =  math.sqrt(2*np.dot(np.dot(w.T,sigma_new),w))
        
        f = lambda y : math.exp(-y**2)
        integ_1 = integrate.quad(f, 0, (dx/dsigma))
        integ_2 = integrate.quad(f, 0, (-dx**2/dsigma**2))
        
        loss_ = (dx/2)*((2/math.sqrt(math.pi))*integ_1[0] + 1) + (dsigma/(2*math.sqrt(math.pi)))*math.exp(-(dx**2)/(dsigma**2))
        loss.append(loss_)
       
      
      loss = np.array(loss)
      cost = self.C/2 * np.linalg.norm(w)**2  + np.mean(loss)
      return loss, cost

    def fit(self, x, t):

        #collect x for predictive step
        self.x = x
        self.t = t

        x_new = self.construct_x_uncertain(x,t)
        
        # checks for labels
        self.classes_ = np.unique(t)
        #t[t==0] = -1

        # initail variables k, w_0
        it = 0
        w = np.ones(len(x)) 
        b = 1               
        #w = np.zeros(len(x[0]))

        obj_func = []
        obj_batchsgd = []
        iter_batchsgd = []
        #sigma = np.identity(len(x[0]))

        for epoch in range(self.max_epochs):
            idx = np.random.permutation(len(t))
            print("Epoch: %d" %(epoch+1), idx)
            #print("Epoch: %d" %(epoch+1))
            for i in range(len(t)):
                
                r = idx[i*self.n_batches:(i+1)*self.n_batches]
                if r.size==0: break

                it = it + 1
                print("----it: ", it)
                iter_batchsgd.append(it)
                print("----Iteration: %d" %(i+1), r)

                x_kernel = self.kernel(x,x)
                # compute cost function
                loss, cost = self.cost_function(w,b,x_kernel,t)
                obj_func.append(cost) 


                X = x[r,:]
                T = t[r]

                K = np.ones((self.n_batches,len(x)))
                K = self.kernel(X,x)

                sigma = self.sigma(K)
                kv = self.kvtest(K,T)
                #print('t=', T)

                X_new = x_new[r,:]
                T_new = t[r]

                #map (x_random, X) with rbf kernel
                K_new = np.ones((self.n_batches,len(x_new)))
                K_new  = self.kernel(X_new,x_new)

                #sigma = self.sigma(K)
                
                # kv = self.kvtest(X,T)
                # print('kv=', kv)
               
                
                # compute gradient of loss depend on w 
                gradloss = self.gradient(w, b, K_new, T_new, sigma, kv)
                gradloss = np.vstack(gradloss)
                gloss = np.mean(gradloss, axis = 0)

                # compute gradient depend on w 
                grad = self.C* w + gloss

                
                # step size
                eta =1/it
                
                # update weight
                w -= eta*grad
                # l2 = norm(w,2)
                # w_ = min(1, (1/math.sqrt(self.C))/l2)*w
                #print("----w: ",w_)
                

                # compute gradient depend on b 
                bgrad1 = self.bgradient(w, b, K_new, T_new, sigma, kv)
                bgrad2 = np.vstack(bgrad1)
                bgrad = (-1)*np.mean(bgrad2)
                         
                # update bias
                b -= eta*bgrad
                print("----b: ",b)
                        
        self.final_iter = it
        self._coef = w
        self._intercept = b
        self.obj_func = obj_func
        self.obj_batchsgd = obj_batchsgd
        self.iter_batchsgd = iter_batchsgd

        return self

    
    def predict(self, z):
        X_norm = np.sum(z ** 2, axis = -1)
        Y_norm = np.sum(self.x ** 2, axis = -1)
        x = ne.evaluate('exp(-g * (A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : Y_norm[None,:],
                'C' : np.dot(z, self.x.T),
                'g' : self.gamma,
        })
        p = np.sign(np.matmul(x,self._coef)+self._intercept)
        p[p==0] = 1
        return p.astype(int)

