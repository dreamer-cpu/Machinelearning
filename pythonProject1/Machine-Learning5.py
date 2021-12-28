import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sio.loadmat('D:\pythonProject1\ex5\ex5data1.mat')
X,y,Xval,yval,Xtest,ytest = map(np.ravel,[data['X'],data['y'],data['Xval'],data['yval'],data['Xtest'],data['ytest']])
#print(X.shape,y.shape,Xval.shape,yval.shape,Xtest.shape,ytest.shape)
'''
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(X,y)
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
'''
#plt.show()

#正则化线性回归代价函数
X,Xval,Xtest = [np.insert(x.reshape(x.shape[0],1),0,np.ones(x.shape[0]),axis=1)for x in (X,Xval,Xtest)]

def cost(theta,X,y):
    m = X.shape[0]
    inner = X@theta-y
    square_sum = inner.T@inner
    cost = square_sum/(2*m)

    return cost
def costReg(theta,X,y,reg=1):
    m = X.shape[0]
    regularized_term = (reg/(2*m))*np.power(theta[1:],2).sum()
    return cost(theta,X,y)+regularized_term

theta = np.ones(X.shape[1])
#print(costReg(theta,X,y,1))

#正则化线性回归的梯度

def gradient(theta,X,y):
    m = X.shape[0]
    inner = X.T@(X@theta-y)

    return inner/m

def gradientReg(theta,X,y,reg):
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0
    regularized_term = (reg/m)*regularized_term

    return gradient(theta,X,y)+regularized_term

#print(gradientReg(theta,X,y,1))

#拟合线性回归

theta = np.ones(X.shape[1])
final_theta = opt.minimize(fun=costReg,x0=theta,args=(X,y,0),method='TNC',jac=gradientReg,options={'disp':True}).x
#print(final_theta)

b = final_theta[0]
m = final_theta[1]
'''
fig,ax = plt.subplots(figsize=(12,8))
plt.scatter(X[:,1],y,c='r',label="Training data")
plt.plot(X[:,1],X[:,1]*m+b,c='b',label="Predicting")
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
ax.legend()
plt.show()
'''

#方差和偏差 偏差大：欠拟合  方差大：过拟合
def linear_regression(X,y,l=1):
    """linear regression
        args:
            X: feature matrix, (m, n+1) # with incercept x0=1
            y: target vector, (m, )
            l: lambda constant for regularization

        return: trained parameters
        """
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=costReg,x0=theta,args=(X,y,l),method='TNC',jac=gradientReg,options={'disp':True})
    return res

training_cost,cv_cost = [],[]
m = X.shape[0]
for i in range(1,m+1):
    res = linear_regression(X[:i,:],y[:i],0)

    tc = costReg(res.x,X[:i,:],y[:i],0)
    cv = costReg(res.x,Xval,yval,0)

    training_cost.append(tc)
    cv_cost.append(cv)
'''
fig,ax = plt.subplots(figsize=(12,8))
plt.plot(np.arange(1,m+1),training_cost,label='training cost')
plt.plot(np.arange(1,m+1),cv_cost,label='cv cost')
plt.legend()
plt.show()
#欠拟合了
'''
