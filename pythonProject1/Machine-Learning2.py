import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'D:\pythonProject1\ex2\ex2data1.txt'
data = pd.read_csv(path,header=None,names=['exam1','exam2','admitted'])
#print(data.head())

positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]
'''
fix,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['exam1'],positive['exam2'],s=50,c='b',marker='o',label='admitted')
ax.scatter(negative['exam1'],negative['exam2'],s=50,c='r',marker='x',label='NOT admitted')
ax.legend()
ax.set_xlabel('exam 1 score')
ax.set_ylabel('exam 2 score')
'''
#plt.show()

def sigmod(z):  #实现sigmod函数
    return 1/(1+np.exp(-z))

def cost(theta,x,y):  #实现代价函数
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmod(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmod(x*theta.T)))
    return np.sum(first-second)/(len(x))

data.insert(0,'Ones',1) #加一列常数列
cols = data.shape[1]
x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)

x = np.array(x.values)
y = np.array(y.values)

#print(x.shape,theta.shape,y.shape)#检查矩阵维度

#print(cost(theta,x,y))

def gradient(theta,x,y): #实现梯度计算
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmod(x*theta.T)-y
    for i in range(parameters):
        term = np.multiply(error,x[:,i])
        grad[i] = np.sum(term)/len(x)
    return grad

#使用工具库计算θ
#result = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(x,y))
#result
#print(result)

#用θ的计算结果带回代价函数计算
#print(cost(result[0],x,y))
'''
plotting_x1 = np.linspace(30,100,100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1,plotting_h1,'y',label='prediction')
ax.scatter(positive['exam1'],positive['exam2'],s=50,c='b',marker='o',label='admitted')
ax.scatter(negative['exam1'],negative['exam2'],s=50,c='r',marker='x',label='not admitted')
ax.legend()
ax.set_xlabel('exam 1 score')
ax.set_ylabel('exam 2 score')
'''
#plt.ylim(20,110)
#plt.show()

def hfunc1(theta,x): #实现h(θ)
    return sigmod(np.dot(theta.T,x))
#print(hfunc1(result[0],[1,45,85])) #测试一个学生exam1=45 exam2=85时的录取概率

#通过模型在训练集上的正确率判断θ
def predict(theta,x): #
    probability = sigmod(x*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]
'''
theta_min = np.matrix(result[0]) #统计预测正确率
predictions = predict(theta_min,x)
correct = [1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(predictions,y)]
accuracy = (sum(map(int,correct))% len(correct))
'''
#print('accuracy ={0}'.format(accuracy))

path1 = 'D:\pythonProject1\ex2\ex2data2.txt'
data_init = pd.read_csv(path1,header=None,names=['test1','test2','accepted'])
#print(data_init.head())

positive2 = data_init[data_init['accepted'].isin([1])]
negative2 = data_init[data_init['accepted'].isin([0])]

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['test1'],positive2['test2'],s=50,c='b',marker='o',label='accepted')
ax.scatter(negative2['test1'],negative2['test2'],s=50,c='r',marker='x',label='rejected')
ax.legend()
ax.set_xlabel('test1 score')
ax.set_ylabel('test2 score')
#plt.show()

degree = 6 #一种更好的使用数据集的方式是为每组数据创造更多的特征,为每组x1，x2添加了最高6次幂的特征
data2 = data_init
x1 = data2['test1']
x2 = data2['test2']

data2.insert(3,'Ones',1)
for i in range(1,degree+1):
    for j in range(0, i + 1):
        data2['F'+str(i-j)+str(j)] = np.power(x1,i-j)*np.power(x2,j)

data2.drop('test1',axis=1,inplace=True)
data2.drop('test2',axis=1,inplace=True)
#print(data2.head())

#实现逻辑回归的代价函数和梯度的函数
def costReg(theta,x,y,learningRate): #实现正则化的代价函数
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmod(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmod(x*theta.T)))
    reg = (learningRate/(2*len(x)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/len(x)+reg

def gradientReg(theta,x,y,learningRate): #实现正则化的梯度函数
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error =sigmod(x*theta.T)-y

    for i in range(parameters):
        term = np.multiply(error,x[:,i])

        if(i==0):
            grad[i]=np.sum(term)/len(x)
        else:
            grad[i]=(np.sum(term)/len(x))+((learningRate/len(x))*theta[:,i])
    return grad

cols1 = data2.shape[1]
x2 = data2.iloc[:,1:cols1]
y2 = data2.iloc[:,0:1]
theta2 = np.zeros(cols1-1)

x2 = np.array(x2.values)
y2 = np.array(y2.values)

learningRate = 1

#print(costReg(theta2,x2,y2,learningRate))

result2 = opt.fmin_tnc(func=costReg,x0=theta2,fprime=gradientReg,args=(x2,y2,learningRate))
#print(result2)

theta_min = np.matrix(result2[0])
predictions = predict(theta_min,x2)
correct = [1 if((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(predictions,y2)]
accuracy = (sum(map(int,correct)) % len(correct))
#print('accuracy = {0}%'.format(accuracy))

#画出决策曲线
def hfunc2(theta,x1,x2):
    temp = theta[0][0]
    place = 0
    for i in range(1,degree+1):
        for j in range(0,i+1):
            temp += np.power(x1,i-j) * np.power(x2,j) * theta[0][place+1]
            place+=1
    return temp
def find_decision_boundary(theta):
    t1 = np.linspace(-1,1.5,1000)
    t2 = np.linspace(-1,1.5,1000)
    cordinates = [(x,y) for x in t1 for y in t2]
    x_cord,y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1':x_cord,'x2':y_cord})
    h_val['hval'] = hfunc2(theta,h_val['x1'],h_val['x2'])
    decision = h_val[np.abs(h_val['hval'])< 2 * 10**-3]
    return  decision.x1,decision.x2
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['test1'],positive2['test2'],s=50,c='b',marker='o',label='accepted')
ax.scatter(negative2['test1'],negative2['test2'],s=50,c='r',marker='x',label='rejected')
ax.set_xlabel('test1 sore')
ax.set_ylabel('test2 score')

x,y = find_decision_boundary(result2)
plt.scatter(x,y,c='y',s=50,label='prediction')
ax.legend()
#plt.show()

#λ=0时过拟合
learningRate2 = 0
result3 = opt.fmin_tnc(func=costReg,x0=theta2,fprime=gradientReg,args=(x2,y2,learningRate2))
#print(result3)

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['test1'],positive2['test2'],s=50,c='b',marker='o',label='accepted')
ax.scatter(negative2['test1'],negative2['test2'],s=50,c='r',marker='x',label='rejected')
ax.set_xlabel('test1 score')
ax.set_ylabel('test2 score')

x,y = find_decision_boundary(result3)
plt.scatter(x,y,c='y',s=10,label='prediction')
ax.legend()
#plt.show()

#λ=100时欠拟合
learningRate3 = 100
result4 = opt.fmin_tnc(func=costReg,x0=theta2,fprime=gradientReg,args=(x2,y2,learningRate3))
fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(positive2['test1'],positive2['test2'],s=50,c='b',marker='o',label='accepted')
ax.scatter(negative2['test1'],negative2['test2'],s=50,c='r',marker='x',label='rejected')
ax.set_xlabel('test1 score')
ax.set_ylabel('test2 score')

x,y =find_decision_boundary(result4)
plt.scatter(x,y,c='y',s=10,label='prediction')
ax.legend()
plt.show()
