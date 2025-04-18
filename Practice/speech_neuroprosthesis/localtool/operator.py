import numpy as np

'''Basic maths'''
## extract and rearrange neural activities
### def Z-score. input a 1d numpy array
def z_score(arr):
    x = np.array(arr.copy())
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


### def subtraction mean. input a 2-2 numpy array. default mean by axis 0
def subt_mean(arr, axis = 0):
    x = np.array(arr.copy())
    mean = np.mean(x, axis = axis)
    if axis == 0:
        return x - mean
    if axis == 1:
        return (x.T - mean).T
    

'''Mahalanobis distance'''
### def average mahalanobis distance of two sets of arrays
def mahalanobis_mean(x, y, cov): # x and y are 2-d matrices in size of (#trials, #channels)
    '''d_M = np.sqrt((x-y)^T \Sigma^{-1} (x-y))
    and return the average of the d_M of all the conditions'''
    # x of size (20, 39)
    # y of size (16, 39)
    mu = np.mean(x, axis = 0) - np.mean(y, axis = 0) # 
    mm = (mu @ np.linalg.solve(cov[0], mu.reshape(len(mu), 1))) + (mu @ np.linalg.solve(cov[1], mu.reshape(len(mu), 1)))
    return mm


### def covariance matrix
def covariance(x, y, lamb = 0.001):
    pcs = np.array(x).shape[1] # actualluy now we only use x; pcs = 39
    xsm = x - np.mean(x, axis = 0) # same size as x (16, 39)
    covx = xsm.T @ xsm # size (39, 39)
    covx += lamb * np.eye(pcs)

    ysm = y - np.mean(y, axis = 0)
    covy = ysm.T @ ysm
    covy += lamb * np.eye(pcs)
    return (covx, covy)


'''similarity'''
'''calculated as the angle between vectors in parameter space'''
## similarity by angle between
### def of cross product calculation
def cross_product(x, y):
    if len(x) != len(y):
        return 0
    
    t = 0
    for i in range(len(x)):
        t += x[i] * y[i]

    return t

### def of representation similarity calculation
def repre_similarity(x, y):
    if cross_product(x, x) == 0 or cross_product(y, y) == 0:
        return 0
    if len(x) != len(y):
        return 0
    
    t = 0
    return cross_product(x, y) / np.sqrt(cross_product(x, x)) / np.sqrt(cross_product(y, y))


'''fake CKA similarity'''
## similarity by (fake-CKA)
### def of centered kernal
def centered_kernal(x): # input variable: 1-d array/list
    n = len(x)
    matGram = np.array(x).reshape(n, 1) @ np.array(x).reshape(1, n)
    matH = np.eye(n) - np.ones((n, n)) / n
    return matH @ matGram @ matH

### def of (fake-)CKA
def ckalike_similarity(x, y):
    tX = centered_kernal(x)
    tY = centered_kernal(y)
    return np.trace(tX @ tY) / np.sqrt(np.trace(tX @ tX)) / np.sqrt(np.trace(tY @ tY))
