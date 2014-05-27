'''
Created on 20.05.2014

@author: Digusil
'''

import numpy as np

def kernelList(restrictiveU):
    if restrictiveU:
        return ['uniform', 'triangle', 'cosinus', 'epanechnikov1', 'epanechnikov2', 'epanechnikov3']
    else:
        return ['gaussian', 'cauchy', 'picard']

def kernel(kernelString):
    if kernelString == 'gaussian':
        return gaussianKernel
    elif kernelString == 'cauchy':
        return cauchyKernel
    elif kernelString == 'picard':
        return picardKernel
    elif kernelString == 'uniform':
        return uniformKernel
    elif kernelString == 'triangle':
        return triangleKernel
    elif kernelString == 'cosinus':
        return cosKernel
    elif kernelString == 'epanechnikov1':
        return epanichnikov1
    elif kernelString == 'epanechnikov2':
        return epanichnikov1
    elif kernelString == 'epanechnikov3':
        return epanichnikov1
    else:
        raise NameError('Kernel function not found! Use a valid kernel function.')
    
def gaussianKernel(u, derivative):
    K = 1 / np.sqrt(2 * np.pi) * np.exp(-np.power(u, 2) / 2)
    if derivative == 0:
        return K
    else:
        dK = -u * K
        
        if derivative == 1:
            return K, dK
        else:
            ddK = (np.power(u, 2) - 1) * K
            
            return K, dK, ddK
        
def cauchyKernel(u, derivative):
    K = 1 / np.sqrt(np.pi * (1+np.power(u, 2)))
    if derivative == 0:
        return K
    else:
        dK = -2*u /(1+np.power(u, 2)) * K
        
        if derivative == 1:
            return K, dK
        else:
            ddK = (-2*np.power(1+np.power(u, 2),2) + 8*np.power(u, 2)) / \
                  (np.pi * np.power(1+np.power(u, 2),3))
            
            return K, dK, ddK
        
def picardKernel(u, derivative):
    K = 1 / 2 * np.exp(-np.abs(u))
    if derivative == 0:
        return K
    else:
        dK = -np.sign(u) * K
        
        if derivative == 1:
            return K, dK
        else:
            ddK = K
            
            return K, dK, ddK
        
def uniformKernel(u, derivative):
    indeces_u = np.abs(u) > 1
    
    K = 1 / 2 * np.ones(u.shape)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = np.zeros(u.shape)
        
        if derivative == 1:
            return K, dK
        else:
            ddK = np.zeros(u.shape)
            
            return K, dK, ddK
        
def triangleKernel(u, derivative):
    indeces_u = np.abs(u) > 1
    
    K = 1 - np.abs(u)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = -np.sign(u)
        dK[indeces_u] = 0
        
        if derivative == 1:
            return K, dK
        else:
            ddK = np.zeros(u.shape)
            
            return K, dK, ddK
        
def cosKernel(u, derivative):
    indeces_u = np.abs(u) > 1
    
    K = np.pi/4 * np.cos(np.pi/2*u)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = -np.power(np.pi,2)/8 * np.sin(np.pi/2*u)
        dK[indeces_u] = 0
        
        if derivative == 1:
            return K, dK
        else:
            ddK = -np.power(np.pi,3)/16 * np.cos(np.pi/2*u)
            ddK[indeces_u] = 0
            return K, dK, ddK
        
def epanechnikovKernel(u, derivative, p):
    indeces_u = np.abs(u) > 1
    
    if p == 1:
        Cp = 3/4
    elif p == 2:
        Cp = 15/16
    elif p == 3:
        Cp = 35/32
    else:
        raise ValueError('Wrong p! Use 1, 2 or 3.')
    
    K = Cp * np.power(1-np.power(u,2),p)
    K[indeces_u] = 0
    
    if derivative == 0:
        return K
    else:
        dK = -2*p*Cp*u*np.power(1-np.power(u,2),p-1)
        dK[indeces_u] = 0
        
        if derivative == 1:
            return K, dK
        else:
            ddK = 2*p*Cp*(2*np.power(u,2)*np.power(1-np.power(u,2),p-2)\
                           -np.power(1-np.power(u,2),p-1))
            ddK[indeces_u] = 0
            return K, dK, ddK
        
def epanichnikov1(u, derivative):
    return epanechnikovKernel(u, derivative, p = 1)

def epanichnikov2(u, derivative):
    return epanechnikovKernel(u, derivative, p = 2)

def epanichnikov3(u, derivative):
    return epanechnikovKernel(u, derivative, p = 3)