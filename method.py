import numpy as np
from numpy import linalg as la
import random
import math

def SGD(cost, grad, hess, K, gamma, x0, batch_size, n):   

    #batches
    batch = []
    for i in range(K): 
        batch.append(random.sample(range(n),batch_size))   
        #生成0~N之间的不重复的指定长度L的随机整数，可以使用:random.sample(range(0, N), L)，例如要生成0~20之间的20个随机整数list：
        #a = random.sample(range(0, 20), 20)
        #>>>a
        #[13, 15, 16, 6, 0, 2, 17, 18, 9, 5, 14, 3, 12, 19, 7, 10, 11, 1, 4, 8]
    ## initialization
    x = [x0 for i in range(K)]#K个初始点x0数组的list：x[0],x[1],..x[K-1]=x0
    f = np.zeros((K,))#数组
    gammas = np.zeros((K-1,))#数组
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        gammas[k] = gamma
        x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
        f[k+1] = cost(x[k+1],range(n))
    name = 'SGD, step='+"{:.2f}".format(gamma)
    return name, f, gammas

def SPS(cost, grad, hess, K, c, gamma_max, x0, batch_size, n):    
        #batches        
    batch = []
    for i in range(K): 
        batch.append(random.sample(range(n),batch_size))
    x = [x0 for i in range(K)]
    f = np.zeros((K,))#数组
    gammas = np.zeros((K-1,))
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        sps_grad = cost(x[k],batch[k])/la.norm(grad(x[k],batch[k]))**2
        gammas[k] = min([sps_grad/c,gamma_max])                   
            # update
        x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
        f[k+1] = cost(x[k+1],range(n))  
    ## name            
    name = 'SPS, step='+"{:.2f}".format(gamma_max)   
    return name, f, gammas

def SPSlack(cost, grad, hess, K, gamma_max, x0,s0, batch_size, n):
    batch = []
    for i in range(K): 
        batch.append(random.sample(range(n),batch_size))
    x = [x0 for i in range(K)]
    s = [s0 for i in range(K)]
    f = np.zeros((K,))#数组
    gammas = np.zeros((K-1,))
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        sps_grad = cost(x[k],batch[k])/la.norm(grad(x[k],batch[k]))**2
        gammas[k] = min([sps_grad,gamma_max])                   
            # update
        s[k+1] = max([ cost(x[k],batch[k])-gamma_max*la.norm(grad(x[k],batch[k]))**2 , 0 ])
        if s[k+1]==0:
            x[k+1] = x[k] - sps_grad*grad(x[k],batch[k])
        else:
            x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
        f[k+1] = cost(x[k+1],range(n))  
    ## name            
    name = 'SPS, step='+"{:.2f}".format(gamma_max)   
    return name, f, gammas
    
def SPSALI(cost, grad, hess, K, gamma_max, x0,s0, batch_size, n):
    batch = []
    for i in range(K): 
        batch.append(random.sample(range(n),batch_size))
    x = [x0 for i in range(K)]
    s = [s0 for i in range(K)]
    f = np.zeros((K,))#数组
    gammas = np.zeros((K-1,))
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        sps_grad = cost(x[k],batch[k])/(la.norm(grad(x[k],batch[k]))**2+1/gamma_max)
        gammas[k] = min([sps_grad,gamma_max])                   
            # update
        s[k+1] = cost(x[k],batch[k])/(1+gamma_max*la.norm(grad(x[k],batch[k]))**2)
        
        x[k+1] = x[k] - sps_grad*grad(x[k],batch[k])
        
        f[k+1] = cost(x[k+1],range(n))  
    ## name            
    name = 'SPS, step='+"{:.2f}".format(gamma_max)   
    return name, f, gammas

def SPSL1(cost, grad, hess, K, gamma_max, x0,s0, batch_size, n):
    batch = []
    for i in range(K): 
        batch.append(random.sample(range(n),batch_size))
    x = [x0 for i in range(K)]
    s = [s0 for i in range(K)]
    f = np.zeros((K,))#数组
    gammas = np.zeros((K-1,))
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        sps_grad = cost(x[k],batch[k])/(la.norm(grad(x[k],batch[k]))**2)
          
        n1=max(cost(x[k],batch[k])-s[k]+gamma_max,0)
        n2=n1/(1+la.norm(grad(x[k],batch[k]))**2)
            # update
        s[k+1] = max(s[k]-gamma_max+n2,0)
        gammas[k] = min([sps_grad,n2])
        x[k+1] = x[k] - gammas[k]*grad(x[k],batch[k])
        
        f[k+1] = cost(x[k+1],range(n))  
    ## name            
    name = 'SPS, step='+"{:.2f}".format(gamma_max)   
    return name, f, gammas

def SPSL2(cost, grad, hess, K, gamma_max, x0,s0, batch_size, n):
    batch = []
    for i in range(K): 
        batch.append(random.sample(range(n),batch_size))
    x = [x0 for i in range(K)]
    s = [s0 for i in range(K)]
    f = np.zeros((K,))#数组
    gammas = np.zeros((K-1,))
    f[0] = cost(x0,range(n))
    for k in range(K-1):
        sps_grad = cost(x[k],batch[k])/(la.norm(grad(x[k],batch[k]))**2)
        gammas[k] = min([sps_grad,gamma_max])
        alpha=1/(1+gamma_max)
        n1=max(cost(x[k],batch[k])-alpha*s[k],0)
        n2=n1/(alpha+la.norm(grad(x[k],batch[k]))**2)                   
            # update
        s[k+1] = alpha*(s[k]+n2)
        
        x[k+1] = x[k] - n2*grad(x[k],batch[k])
        
        f[k+1] = cost(x[k+1],range(n))  
    ## name            
    name = 'SPS, step='+"{:.2f}".format(gamma_max)   
    return name, f, gammas

