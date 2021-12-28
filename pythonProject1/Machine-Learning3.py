import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report#这个包是评价报告
from scipy.optimize import minimize

data = loadmat('D:\pythonProject1\ex3\ex3data1.mat')
#print(data)
#print(data['X'].shape,data['y'].shape)

#随机展示100个数据
sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100)
sample_images = data['X'][sample_idx,:]
#print(sample_images)
'''
fig,ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
'''
#plt.show()

def sigmod(z): #逻辑回归模型的假设函数
    return 1/(1+np.exp(-z))

def cost(theta,x,y,LearningRate): #代价函数
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmod(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmod(x*theta.T)))
    reg = (LearningRate/(2*len(x)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/len(x)+reg
def gradient(theta,x,y,LearningRate): #梯度函数
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    error = sigmod(x*theta.T)-y
    grad = ((x.T*error)/len(x)).T + ((LearningRate/len(x))*theta)
    grad[0,0] = np.sum(np.multiply(error,x[:,0]))/len(x)
    return  np.array(grad).ravel()

#构建分类器
def one_vs_all(x,y,num_labels,learning_rate):
    rows = x.shape[0]
    params = x.shape[1]
    all_theta = np.zeros((num_labels,params+1))
    x = np.insert(x,0,values=np.ones(rows),axis=1)#插入第一列全部是1
    for i in range(1,num_labels+1):
        theta = np.zeros(params+1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i,(rows,1))
        #最小化每个分类器的代价函数
        fmin = minimize(fun=cost,x0=theta,args=(x,y_i,learning_rate),method='TNC',jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta

rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10,params+1))
x = np.insert(data['X'],0,values=np.ones(rows),axis=1)
theta = np.zeros(params+1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0,(rows,1))

#print(x.shape,y_0.shape,theta.shape,all_theta.shape)
#print(np.unique(data['y']))#查看有几类标签

all_theta = one_vs_all(data['X'],data['y'],10,1)
#print(all_theta)

#用训练完毕的分类器预测每个图像的标签
def predict_all(x,all_theta):
    rows = x.shape[0]
    params = x.shape[1]
    num_labels = all_theta.shape[0]
    x = np.insert(x,0,values=np.ones(rows),axis=1)
    x = np.matrix(x)
    all_theta = np.matrix(all_theta)
    h = sigmod(x*all_theta.T)
    h_argmax = np.argmax(h,axis=1)
    h_argmax = h_argmax+1
    return h_argmax

y_pred = predict_all(data['X'],all_theta)
#print(classification_report(data['y'],y_pred))

weight = loadmat("D:\pythonProject1\ex3\ex3weights.mat")
theta1,theta2 = weight['Theta1'],weight['Theta2']
#print(theta1.shape,theta2.shape)
x2 = np.matrix(np.insert(data['X'],0,values=np.ones(x.shape[0]),axis=1))
y2 = np.matrix(data['y'])
#print(x2.shape,y2.shape)

a1 = x2
z2 = a1*theta1.T
#print(z2.shape)

a2 = sigmod(z2)
#a2.shape

a2 = np.insert(a2,0,values=np.ones(a2.shape[0]),axis=1)
z3 = a2*theta2.T
#z3.shape

a3 = sigmod(z3)
y_pred2 = np.argmax(a3,axis=1)+1
print(classification_report(y2,y_pred))


