""" 
Created on Mon May  8 11:58:01 2017 
 
@author: lizhongzheng 

"""  
  
import numpy as np  
import matplotlib.pyplot as plt  
  
from keras.models import Sequential  
from keras.layers import Dense, Activation  
from keras.optimizers import SGD

plt.ion()

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

def makeup_1layer(X, Y, Cy):
    # input data:  X ~ N x Dx,    Y ~ N x 1 each in cardinality of Cy
    # return [v, b] as the parameter ready for model.set_weights()
    
    (N, Dx) = X.shape
    
    Mreg = np.zeros([Dx, Cy])    #conditional mean 
    Py_emp = np.zeros(Cy)      # empirical distribution
    
    # for each y=j, compute empirical average E[X|Y=j]
    for j in range(Cy):
        Mreg[:,j] = sum(X[Y==j])/sum(Y==j)
        Py_emp[j] = sum(Y==j)/N

    Sigma2 = sum(sum(X*X))/N/Dx    # unconditional variance
    # (v, b) generate from UNconditonal variance    
    V = Mreg/Sigma2
    b = np.log(Py_emp) + sum(-(Mreg*Mreg)/Sigma2/2)     
    
    return([V,b])

# the dimensionality of x vectors   
Dx = 2  
  
# the cardinality of y   
Cy = 8  
  
# pick Py randomly, each entry from 1..5, then normalize  
Py = np.random.choice(range(1,4), Cy)  
Py = Py/sum(Py)  
  
# pick the A parameters   
  
A = np.random.uniform(0, 10, [Dx+1, Cy])  
  
# number of samples   
N = 10000  
  
# Generate the samples  
  
Y = np.random.choice(Cy, N, p=Py)  
T = np.zeros([N, Dx+1])  
  
for j in range(Cy):  
    T[Y==j, :] = np.random.dirichlet(A[:, j], sum(Y==j))  
  
X = T[:, :-1]  
  
# make the labels  
Labels = np.zeros([N, Cy]) # neural network takes the indicators instead of Y      
for i in range(N):  
    Labels[i, Y[i]] = 1  
  
# centralize       
Xbar = sum(X)/N  
X = X - np.tile(Xbar, [N,1])

# empirical distribution of Y, we are not supposed to know Py anyway  
PPy = sum(Labels)/N  

# compute the empirical means M[j, :] = E[ X | Y=j ] 
M = np.zeros([Cy, Dx])  
for j in range(Cy):  
    M[j,:] = sum(X[Y==j, :])/sum(Y==j)

model = Sequential()  
model.add(Dense(Cy, activation='softmax', input_dim=Dx))  
  
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(loss='categorical_crossentropy',  optimizer=sgd,  metrics=['accuracy'])  
model.fit(X,Labels, verbose=0, epochs=200, batch_size=200)

plt.close('all')
for i in range(Dx):    
    plt.figure(i)  
    plt.plot(range(Cy), regulate(M[:, i], PPy), 'r-')  
    plt.plot(range(Cy), regulate(model.get_weights()[0][i], PPy), 'b-')  
    plt.title(i)  
    plt.show()
    

nmodel = Sequential() 
nmodel.add(Dense(Cy, activation='softmax', input_dim=Dx))
weights = makeup_1layer(X, Y, Cy) #compute the weights from the data
nmodel.set_weights(weights)

b0 = model.predict_classes(X)
print('\n The result of trained NN', sum(b0==Y)/N)

# check how good it is    
b = nmodel.predict_classes(X)
print('\n The result of my own NN ', sum(b==Y)/N, '\n')
