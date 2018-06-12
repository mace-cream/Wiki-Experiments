#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:43:28 2017

@author: lizhong
"""

"""
Continuous X, Y keras test

"""

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit # calculate Logistic sigmoid

plt.ion()

def Generate2DSamples(xCard, yCard, nSamples):
    
    # randomly pick joint distribution, normalize
    Pxy = np.random.random([xCard, yCard])
    Pxy = Pxy / sum(sum(Pxy))

    # compute marginals
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0)    
    
    lp = np.reshape(Pxy, xCard*yCard)
        
    data = np.random.choice(range(xCard*yCard), nSamples, p=lp)
    
    X = (data/yCard).astype(np.int) 
    Y = data % yCard
    
    return([Pxy, Px, Py, X, Y])
    
def MakeLabels(X):
    '''
    return onehotencoded array
    
    Input X should be an array (nSamples, 1), taking values from alphabet 
    with size xCard
    
    X must be labels themselves, i.e. integers starting from 0, with every 
    value used. Otherwise we need to use sklearn.LabelEncoder()
    
    return array of size (nSamples, xCard)
    '''    
    onehot_encoder = OneHotEncoder(sparse=False)
    temp = X.reshape(len(X), 1)
    onehots = onehot_encoder.fit_transform(temp)
    return(onehots)
    
def regulate(x, p):  
    ''' 
    regulate x vector so it has zero mean and unit variance w.r.t. p 
    '''  
    assert(np.isclose(sum(p),1)), 'invalid distribution used in regulate()'  
    assert(x.size==p.size), 'dimension mismatch in regulate()'  
     
    x = x.reshape(x.size)
    r = x - sum(x*p)  
    r = r / np.sqrt(sum(p*r*r))  
    return(r)
       
def WhatTheorySays(Pxy, weights, PreProcessing):
    
    '''
    Compute theoretical answers for f , corresponding to the 1st pair
    of singular vectors
    Return in normalized function form
    '''
    # compute marginals
    [xCard, yCard] = Pxy.shape
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0) 
    temp = np.tile(np.sqrt(Px).reshape(xCard,1), [1, yCard]) \
        * np.tile(np.sqrt(Py), [xCard, 1])
    B = Pxy / temp
    U, s, V = np.linalg.svd(B)
    # check bias on the output layer
    
    b = weights[3]
    b = b - sum(b*Py)
    
    w = weights[0].reshape(weights[0].size)
    c = weights[1]   
    S = expit(np.matmul(PreProcessing, w) + c) # output of hidden node
    mu_s = sum (S * Px)  # compute E[S(X)]
    
    v = weights[2][0]   
    
    # log(P_Y) - v(y) * E[S(X)]
    b_theory = np.log(Py) - v*mu_s  
    b_theory = b_theory - sum(b_theory*Py)
    
    plt.clf()
    plt.figure(1)
    plt.plot(range(yCard), b, 'r', label="Training Result")
    plt.plot(range(yCard), b_theory, 'b', label="Theoretic")
    plt.legend(loc='lower left')
    plt.title('Output Layer Bias Check')
    plt.show()
    
    # Check weights 

    vtilde = v - sum(v*Py)   # force to be zero mean
    
    psi = vtilde * np.sqrt(Py)
    varg = sum (psi*psi)    # Var[V(Y)]
    
    phi = np.matmul(B, psi)
    f = phi / np.sqrt(Px)   # desired output of hidden layer
                            # f(x) = E [\tilde{V}(Y)|X=x]
    
    gap = f - varg*(S-mu_s)   # difference between desired f and \tilde{S}
    
    derv = np.matmul(np.diag(S*(1-S)), PreProcessing)
    # compute average derivative 
    mean_derv= sum(np.matmul(np.diag(Px), derv))

    # subtract from derivative
    derv = derv - np.tile(mean_derv, [xCard, 1])
    
    orthogonality = np.matmul(np.matmul(np.diag(Px), gap), derv) 

    print('orthogonality check:')
    
    for i in range(PreProcessing.shape[1]):
        a=np.sqrt(Px)*gap
        b=np.sqrt(Px)*derv[:,i]
        print('  ', np.arccos(sum(a*b)/np.sqrt(sum(a*a)*sum(b*b)))*180/np.pi)

xCard = 8
yCard = 6

tDim = 4

nSamples = 100000

[Pxy, Px, Py, X, Y] = Generate2DSamples(xCard, yCard, nSamples)

XLabels = MakeLabels(X)
YLabels = MakeLabels(Y)

# randomly choose a PreProcessing function
PreProcessing = np.random.normal(size=[xCard, tDim])

# make sure functions have zero mean w.r.t. Px
mean = sum(np.matmul(np.diag(Px), PreProcessing))
PreProcessing = PreProcessing - np.tile(mean, [xCard, 1])

# Apply the PreProcessing
T = np.matmul(XLabels, PreProcessing)


model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=tDim))
model.add(Dense(yCard, activation='softmax', input_dim=1))

sgd = SGD(4, decay=1e-2, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(T, YLabels, verbose=0, batch_size=nSamples, epochs=100) 

weights = model.get_weights()

WhatTheorySays(Pxy, weights, PreProcessing)


