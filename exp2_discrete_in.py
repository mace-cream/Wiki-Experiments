"""
Created on Thu Nov  9 10:43:28 2017

@author: lizhong, modified by xiangxiang
"""
  
import numpy as np  
import matplotlib.pyplot as plt  
  
from keras.models import Sequential  
from keras.layers import Dense, Activation  
from keras.optimizers import SGD

from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit # calculate Logistic sigmoid

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
    onehot_encoder = OneHotEncoder(sparse=False)
    temp = X.reshape(len(X), 1)
    onehots = onehot_encoder.fit_transform(temp)  # array [len(x), xCard]
    return(onehots)

xCard = 8
yCard = 6
nSamples = 100000


def WhatTheorySays(Pxy):
    '''
    Compute theoretical answers for f and g, corresponding to the 1st pair
    of singular vectors
    Return in normalized function form
    '''
    # compute marginals
    [xCard, yCard] = Pxy.shape
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0) 
    
    T = np.tile(np.sqrt(Px).reshape(xCard,1), [1, yCard]) \
        * np.tile(np.sqrt(Py), [xCard, 1])
    B = Pxy / T
    P, d, Q = np.linalg.svd(B)
    
    S = P[:,1] / np.sqrt(Px)
    v = Q[1,:] / np.sqrt(Py)
    
    return([S, v])

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

[Pxy, Px, Py, X, Y] = Generate2DSamples(xCard, yCard, nSamples)

[S_theory, v_theory] = WhatTheorySays(Pxy)


XLabels = MakeLabels(X)
YLabels = MakeLabels(Y)

model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim=xCard))
model.add(Dense(yCard, activation='softmax', input_dim=1))

sgd = SGD(4, decay=1e-2, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(XLabels, YLabels, verbose=0, batch_size=nSamples, epochs=100) 

weights = model.get_weights()

S = weights[0].reshape(1, xCard)
S = expit(S + weights[1])
S = regulate(S, Px)
S = S*np.sign(sum(S*S_theory))    # make sure there is no (-1) factor
v = weights[2]
v = regulate(v, Py) 
v = v*np.sign(sum(v*v_theory))

plt.figure()
plt.subplot('121')
plt.plot(range(xCard), S,  'r', label='Training Result')
plt.plot(range(xCard), S_theory, 'b', label='Theoretic')
plt.legend(loc='lower left')
plt.title('S(x): The selected feature function')
plt.subplot('122')
plt.plot(range(yCard), v, 'r', label='Training Result')
plt.plot(range(yCard), v_theory, 'b', label='Theoretic')
plt.legend(loc='lower left')
plt.title('v(y): The output layer weights')
plt.show()


