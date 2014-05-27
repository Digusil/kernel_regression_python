'''
Created on 16.05.2014

@author: Digusil
'''

import pickle
import numpy as np
from scipy.optimize import minimize
from kernel_regression_python.kernelfunctions import kernel, kernelList
import time
import pylab as pl

def krFeature(x, x_feature, powerList=[1]):
    m_1 = np.shape(x)
    m_2 = np.shape(x_feature)
    
    x2 = np.sum(np.power(x, 2), axis=1).reshape((m_1[0], 1))
    X2 = np.sum(np.power(x_feature, 2), axis=1).reshape((1, m_2[0]))
    
    u_feature = np.power(np.sqrt(np.abs(x2 + X2 - 2 * np.dot(x, x_feature.T))), powerList[0])
    
    if len(powerList) > 1:
        for k in range(len(powerList) - 1):
            tmp = np.power(np.sqrt(np.abs(x2 + X2 - 2 * np.dot(x, x_feature.T))), powerList[k + 1])
            u_feature = np.append(u_feature, tmp, axis=1)

    return u_feature

def nadarayaWatsonEstomator(u_feature, y_feature, kernelFunction, h, scaleKernel=True, \
                            derivative=0):
    m_u = u_feature.shape
    
    u = u_feature / h
    
    if derivative == 0:
        K = kernelFunction(u, derivative)
    elif derivative == 1:
        K, dK = kernelFunction(u, derivative)
    else:
        K, dK, ddK = kernelFunction(u, derivative)
    
    if scaleKernel:
        Kh = K / h

        a = np.dot(Kh, y_feature)
        b = np.sum(Kh, axis=1)
    else:
        a = np.dot(K, y_feature)
        b = np.sum(K, axis=1)
    
    m = a / b
    
    if np.any(np.isnan(m)):
        if scaleKernel:
            m[np.isnan(m)] = np.sum(y_feature / h) / np.sum(1 / h)
        else:
            m[np.isnan(m)] = np.sum(y_feature)
            
    if derivative == 0:
        return m
    else:        
        if scaleKernel:
            dKh = -(K + u * dK) / np.power(h, 2)
            da = dKh * y_feature
            db = dKh
        else:
            dK = -u / h * dK
            da = dK * y_feature
            db = dK
            
        if h.size == 1:
            da = np.sum(da, axis=1)
            db = np.sum(db, axis=1)
        else:
            a = a.reshape((m_u[0], 1))
            b = b.reshape((m_u[0], 1))
        
        dm = da / b - a * db / np.power(b, 2)
        
        if np.any(np.isnan(dm)):
            dm[np.isnan(dm)] = 0
            
        if np.any(np.isinf(dm)):
            dm[np.isinf(dm)] = 0
        
        if derivative == 1:
            return m, dm
        else:
            if scaleKernel:
                ddKh = -(u * dK + K) / np.power(h, 3) - u / h * (dKh + u * ddK)
                dda = ddKh * y_feature
                ddb = ddKh
            else:
                ddK = u / np.power(h, 2) * (2 * dK + u * ddK)
                dda = ddK * y_feature
                ddb = ddK
                
            if h.size == 1:
                dda = np.sum(dda, axis=1)
                ddb = np.sum(ddb, axis=1)
                
            ddm = (dda * b - 2 * da * db - a * ddb) / np.power(b, 2) + \
                  (a * np.power(db, 2) / np.power(b, 3))
            
            if np.any(np.isnan(ddm)):
                if scaleKernel:
                    ddm[np.isnan(ddm)] = -np.sum(y_feature / h) / np.sum(1 / h)
                else:
                    ddm[np.isnan(ddm)] = -np.sum(y_feature)
            
            return m, dm, ddm
        
def krCostFunction(u_feature, y_val, y_feature, kernelFunction, h, scaleMode):
    m_u = u_feature.shape
    
    m, dm = nadarayaWatsonEstomator(u_feature, y_feature, kernelFunction, h, scaleMode, 1)
    
    hypo = m - y_val
    
    J = np.dot(hypo, hypo) / (2 * m_u[0])
    
    dJ = np.dot(hypo, dm) / m_u[0]
    
    if np.isscalar(J):
        J = np.asarray([J])
        
    if np.isscalar(dJ):
        dJ = np.asarray([dJ])
    
    return J, dJ

def estimateH(x, rho=False):
    m_x = x.shape
    
    u = krFeature(x, x)
    
    h_est = 0.22 * np.power(m_x[0], -0.185) * np.power(m_x[1], 0.45) * np.median(u)
    
    if rho:
        u[np.isnan(u)] = np.inf
        rho = np.power(np.min(np.abs(u)), m_x[1]) / m_x[0]
        return h_est, rho
    else:
        return h_est

def learnKernelRegression(x_val, y_val, x_feature, y_feature, options):
    u_feature = krFeature(x_val, x_feature, powerList=options['powerList'])
    
    if options['kernel'] in kernelList(restrictiveU=False):
        first_h = estimateH(x_feature)
    elif options['kernel'] in kernelList(restrictiveU=True):
        first_h = np.max(u_feature)
    else:
        raise NameError('Kernel function not found! Use a valid kernel function.')
    
    y_featureL = y_feature
    
    if len(options['powerList']) > 1:
        for k in range(len(options['powerList']) - 1):
            
            y_featureL = np.append(y_featureL, y_feature)
            
    if options['multiH']:
        initial_h = first_h * np.ones(y_featureL.size)
    else:
        initial_h = np.asarray([first_h])
    
    minimizerOptions = {'disp':False}
    
    for k in options:
        minimizerOptions.update({k:options[k]})
        
    del minimizerOptions['kernel']
    del minimizerOptions['multiH']
    del minimizerOptions['scaleKernel']
    del minimizerOptions['reduceFeature']
    del minimizerOptions['powerList']
    
    def minFun(x):
        return krCostFunction(u_feature, y_val, y_featureL, kernel(options['kernel']), x, \
                              options['scaleKernel'])
    
    if not options['disp']:
        print('Optimization is running...')
        t = time.clock()
        h_opt = minimize(minFun, initial_h, method='L-BFGS-B', jac=True, options=minimizerOptions)
        t = time.clock() - t
        print('Optimization end after {0:.2e} seconds'.format(t))
    else:
        h_opt = minimize(minFun, initial_h, method='L-BFGS-B', jac=True, options=minimizerOptions)
    
    return h_opt

def goodnessOfFit(y_appr, y_true):
    y_mean = np.nanmean(y_true)
    
    tmp1 = y_true - y_appr
    tmp2 = y_true - y_mean
    
    r2 = 1 - np.dot(tmp1, tmp1) / np.dot(tmp2, tmp2)   
    
    return r2

def reduceFeature(N, x_feature, y_feature):
    if N < 0 or not np.isscalar(N):
        raise ValueError('Wrong N!')
    
    m_feature = x_feature.shape
    
    h_est, rho = estimateH(x_feature, rho=True)
    
    m = nadarayaWatsonEstomator(krFeature(x_feature, x_feature), y_feature, \
                                         kernel('gaussian'), h_est, 'scaled', 2)
    
    para = np.abs(np.sum(np.outer(m[2], m[2]), axis=1)) * rho
    
    idx = np.argsort(para)
    
    if N < 1:
        idxMin = np.round(N * (m_feature[0] - 1))
    else:
        idxMin = N
            
    if idxMin >= m_feature[0]:
        return np.array([])
    else:
        return np.sort(idx[idxMin:])

def saveData(obj, filename):
    with open(filename, 'wb') as outputData:
        pickle.dump(obj, outputData)
    
def loadData(filename):
    with open(filename, 'rb') as inputData:
        return pickle.load(inputData)
    
def plotRegression(y_appr, y_true):
    r2 = goodnessOfFit(y_appr, y_true)
    
    min_screen = np.min(np.append(y_appr, y_true))
    max_screen = np.max(np.append(y_appr, y_true))
    
    pl.plot(np.array((min_screen, max_screen)), np.array((min_screen, max_screen)), \
            color="black", linewidth=1.0, linestyle="-")
    pl.axis('equal')
    pl.axis([min_screen, max_screen, min_screen, max_screen])
    
    pl.scatter(y_true, y_appr)
    
    pl.title('goodness of fit: {0:.3f}'.format(r2))
    
    pl.show()
    
class Data(object):
    def __init__(self, inputData, targetData, name=[]):
        self.__input = inputData
        self.__target = targetData
        self.__name = name
        
        self.__nExaples = inputData.shape[0]
        self.__nFeatures = inputData.shape[1]
        self.__nDim = targetData.shape[1]
        
    def getInput(self, idx='all'):
        if idx == 'all':
            return self.__input
        else:
            return self.__input[:, idx]
    
    def getTarget(self, idy='all'):
        if idy == 'all':
            return self.__target
        else:
            return self.__target[:, idy]
    
    def setInput(self, values):
        self.__input = values
        self.__nExaples = values.shape[0]
        self.__nFeatures = values.shape[1]
        
    def setTarget(self, values):
        self.__target = values
        self.__nDim = values.shape[1]
        
    def setName(self, aString):
        self.__name = aString
        
    def nExamples(self):
        return self.__nExaples
    
    def normalizeInput(self, mu, sigma, idx='all'):
        if idx == 'all':
            return (self.__input - mu) / sigma
        else:
            return (self.__input[:, idx] - mu[idx]) / sigma[idx]
        
    def normalizeTarget(self, mu, sigma, idy='all'):
        if idy == 'all':
            return (self.__target - mu) / sigma
        else:
            return (self.__target[:, idy] - mu[idy]) / sigma[idy]
                
class KrDataSet(object):
    def __init__(self, inputData, targetData, distribution=(60, 20, 20), \
                 nameStrings=('feature', 'validate', 'test')):        
        self.__nameStrings = nameStrings
        self.__distribution = distribution
        
        self.__data = self.splitData(Data(inputData, targetData))
        
        self.__xMu = np.nanmean(inputData, axis=0)
        self.__xSigma = np.nanstd(inputData, axis=0)
        
        self.__yMu = np.nanmean(targetData, axis=0)
        self.__ySigma = np.nanstd(targetData, axis=0)
        
        self.__nTarget = self.__yMu.size
        
        self.__reduceIndeces = []
    
    def splitData(self, data):
        indeces_perm = np.random.permutation(data.nExamples())
        
        splitData = []
        
        ind = np.array((0, -1))
        
        for k in range(len(self.__distribution) - 1):
            ind = np.array(range(ind[-1] + 1, ind[-1] + 1 + \
                                 np.round(data.nExamples() * self.__distribution[k] \
                                          / np.sum(self.__distribution), 0)))
            tmpInput = data.getInput()
            tmpTarget = data.getTarget()
            splitData.append(Data(tmpInput[indeces_perm[ind]], tmpTarget[indeces_perm[ind]]))
            
        ind = np.array(range(ind[-1], data.nExamples() - 1))
        splitData.append(Data(data.getInput()[indeces_perm[ind]], \
                              data.getTarget()[indeces_perm[ind]]))
        
        if len(self.__nameStrings) > 0:
            for k in range(len(self.__distribution)):
                splitData[k].setName(self.__nameStrings[k])
                
        return splitData
        
    def getDataID(self, aString):
        for k in range(len(self.__nameStrings)):
            if self.__nameStrings[k] == aString:
                return k
    
    def getData(self, dataset, normalize=False):
        if normalize:
            return self.__data[dataset].normalizeInput(mu=self.__xMu, sigma=self.__xSigma), \
                   self.__data[dataset].normalizeTarget(mu=self.__yMu, sigma=self.__ySigma)
        else:
            return self.__data[dataset].getInput(), \
                   self.__data[dataset].getTarget()
    
    def __getReduceIndex(self, i_target, N):
        return reduceFeature(N, self.__data[0].getInput(), self.__data[0].getTarget(i_target))
    
    def reduceFeature(self, N):
        self.__reduceIndeces = []
        
        if np.isscalar(N):        
            for k in range(self.__nTarget):
                self.__reduceIndeces.append(self.__getReduceIndex(k, N))
        else:
            for k in range(self.__nTarget):
                self.__reduceIndeces.append(self.__getReduceIndex(k, N[k]))
            
    def getInput(self, dataset=0, i_target=0, normalize=False, reduceData=False, \
                 idx='all'):
        if dataset == 'all':
            dataset = range(len(self.__distribution))
            
        if np.isscalar(dataset):
            if normalize:
                unreduced = self.__data[dataset].normalizeInput(mu=self.__xMu, \
                                                                sigma=self.__xSigma, \
                                                                idx=idx)
            else:
                unreduced = self.__data[dataset].getInput(idx=idx)
                
            if reduceData:
                return unreduced[self.__reduceIndeces[i_target]]
            else:
                return unreduced
                
        else:
            if normalize:
                unreduced = self.__data[dataset[0]].normalizeInput(mu=self.__xMu, \
                                                                   sigma=self.__xSigma, \
                                                                   idx=idx)
            else:
                unreduced = self.__data[dataset[0]].getInput(idx=idx)
                
            for k in dataset[1:]:
                if normalize:
                    unreduced = np.append(unreduced, \
                                          self.__data[dataset[k]].normalizeInput(\
                                                                             mu=self.__xMu, \
                                                                             sigma=self.__xSigma, \
                                                                             idx=idx), axis=0)
                else:
                    unreduced = np.append(unreduced, self.__data[dataset[k]].getInput(idx=idx), \
                                          axis=0)
                    
            return unreduced
    
    def getTarget(self, dataset=0, i_target=0, normalize=False, reduceData=False):
        if dataset == 'all':
            dataset = range(len(self.__distribution))
        
        if np.isscalar(dataset):    
            if normalize:
                unreduced = self.__data[dataset].normalizeTarget(mu=self.__yMu, \
                                                                 sigma=self.__ySigma, \
                                                                 idy=i_target)
            else:
                unreduced = self.__data[dataset].getTarget(i_target)
                
            if reduceData:
                return unreduced[self.__reduceIndeces[i_target]]
            else:
                return unreduced
            
        else:
            if normalize:
                unreduced = self.__data[dataset[0]].normalizeTarget(mu=self.__yMu, \
                                                                    sigma=self.__ySigma, \
                                                                    idy=i_target)
            else:
                unreduced = self.__data[dataset[0]].getTarget(i_target)
                
            for k in dataset[1:]:
                if normalize:
                    unreduced = np.append(unreduced, \
                                          self.__data[dataset[k]].normalizeTarget(\
                                                                    mu=self.__yMu, \
                                                                    sigma=self.__ySigma, \
                                                                    idy=i_target), axis=0)
                else:
                    unreduced = np.append(unreduced, \
                                          self.__data[dataset[k]].getTarget(i_target), axis=0)
                    
            return unreduced
        
    def getNormalisationData(self):
        return self.__xMu, self.__xSigma, self.__yMu, self.__ySigma
    
    def nTarget(self):
        return self.__nTarget
    
    def splitDistribution(self):
        return self.__distribution
    
    def saveDataset(self, filename):
        saveData(self, filename)
        print('Datase is saved to ' + filename)
    
class KRModell(object):
    def __init__(self, krData=[], options={'multiH':False, 'scaleKernel':True, \
                 'kernel':'gaussian', 'powerList':[1]}):
        self.__krData = krData
        self.__h = []
        self.__r2 = []
        self.__options = {'multiH':False, 'scaleKernel':True, \
                 'kernel':'gaussian', 'powerList':[1]}
        
        self.__options.update(options)
        
    def setData(self, data):
        self.__krData = data
        
    def setOptions(self, options):
        self.__h = []
        self.__r2 = []
        
        self.__options.update(options)
            
    def getData(self):
        return self.__krData
    
    def getOptions(self):
        return self.__options
    
    def h(self):
        return self.__h
    
    def r2(self):
        return self.__r2
    
    def learnModell(self, options = {}):
        h = []
        
        self.__options.update(options)
        
        for i_target in range(self.__krData.nTarget()):
            h.append(learnKernelRegression(self.__krData.getInput(dataset=1, \
                                                                  reduceData=False, \
                                                                  normalize=True, \
                                                                  i_target=i_target), \
                                           self.__krData.getTarget(dataset=1, \
                                                                   reduceData=False, \
                                                                   normalize=True, \
                                                                   i_target=i_target), \
                                           self.__krData.getInput(dataset=0, \
                                                                  reduceData=\
                                                                  self.__options['reduceFeature'], \
                                                                  normalize=True, \
                                                                  i_target=i_target), \
                                           self.__krData.getTarget(dataset=0, \
                                                                  reduceData=\
                                                                  self.__options['reduceFeature'], \
                                                                  normalize=True, \
                                                                  i_target=i_target), \
                                           self.__options))
        self.__h = h
        self.__r2 = self.goodnessOfFit(i_target='all')
        
    def estimateFunction(self, x, normalInput=False, normalOutput=False, i_target='all'):
        normData = self.__krData.getNormalisationData()
        
        if normalInput:
            x = (x - normData[0]) / normData[1]
        
        if i_target == 'all':
            i_target = range(self.__krData.nTarget())
            
        if np.isscalar(i_target):
            y_feature = self.__krData.getTarget(dataset=0, \
                                                reduceData=self.__options['reduceFeature'], \
                                                normalize=True, i_target=i_target)
            
            y_featureL = y_feature
            
            if len(self.__options['powerList']) > 1:
                    for _ in range(len(self.__options['powerList']) - 1):
                        y_featureL = np.append(y_featureL, y_feature)
            
            y = nadarayaWatsonEstomator(krFeature(x, \
                              self.__krData.getInput(dataset=0, \
                                                     reduceData=self.__options['reduceFeature'], \
                                                     normalize=True, i_target=i_target)\
                                                     , powerList=self.__options['powerList']), \
                                                     y_featureL, \
                                                     kernel(self.__options['kernel']), \
                                                     self.__h[i_target].x, \
                                                     self.__options['scaleKernel'], 0)
        else: 
            y = np.zeros([x.shape[0], len(i_target)])
        
            for idy in i_target:
                y_feature = self.__krData.getTarget(dataset=0, \
                                                    reduceData=self.__options['reduceFeature'], \
                                                    normalize=True, i_target=idy)
                
                y_featureL = y_feature
                
                if len(self.__options['powerList']) > 1:
                    for _ in range(len(self.__options['powerList']) - 1):
                        y_featureL = np.append(y_featureL, y_feature)
                
                y[:, idy] = nadarayaWatsonEstomator(krFeature(x, \
                              self.__krData.getInput(dataset=0, \
                                                     reduceData=self.__options['reduceFeature'], \
                                                     normalize=True, i_target=idy)\
                              , powerList=self.__options['powerList']), \
                              y_featureL, \
                              kernel(self.__options['kernel']), \
                              self.__h[idy].x, \
                              self.__options['scaleKernel'], 0)
        if normalOutput:
            return y
        else:
            return y * normData[3][i_target] + normData[2][i_target]

    def saveModell(self, filename):
        saveData(self, filename)
        print('KRModell is saved to ' + filename)
        
    def loadDataset(self, filename):
        self.__h = []
        
        self.__krData = loadData(filename)
        print('Dataset is loaded from ' + filename)
        
    def goodnessOfFit(self, i_target='all'):
        a_test = self.estimateFunction(self.__krData.getInput(dataset=2))
        
        if i_target == 'all':
            r2 = []
            for i_target in range(self.__krData.nTarget()):
                r2.append(goodnessOfFit(a_test[:, i_target], \
                                        self.__krData.getTarget(dataset=2, i_target=i_target)))
        else:
            r2 = goodnessOfFit(a_test[:, i_target], self.__krData.getTarget(dataset=2, \
                                                                            i_target=i_target))
            
        return r2
    
    def plotRegression(self, dataset='all', i_target='all'):
        if i_target == 'all':
            i_target = range(self.__krData.nTarget())
            
        if dataset == 'all':
            dataset = range(len(self.__krData.splitDistribution()))
            
        if np.isscalar(i_target):
            plotRegression(self.estimateFunction(self.__krData.getInput(dataset=dataset), \
                                                 i_target=i_target), \
                           self.__krData.getTarget(dataset=dataset, i_target=i_target))
            
        else:
            y_appr = self.estimateFunction(self.__krData.getInput(dataset=dataset), \
                                           i_target=i_target[0])
            y_true = self.__krData.getTarget(dataset=dataset, i_target=i_target[0])
            
            for k in i_target[1:]:
                y_appr = np.append(y_appr, \
                                   self.estimateFunction(self.__krData.getInput(dataset=dataset), \
                                                         i_target=i_target[k]))
                y_true = np.append(y_true, self.__krData.getTarget(dataset=dataset, \
                                                                   i_target=i_target[k]))  
            
            plotRegression(y_appr, y_true)
            
            
            
