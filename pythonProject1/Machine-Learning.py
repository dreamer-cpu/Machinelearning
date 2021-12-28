import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A = np.eye(5)  #5*5矩阵
# print(A)

path = 'D:\pytest\ex1data1.txt' #文件路径
data = pd.read_csv(path, header=None, names=['Population', 'Profit']) #读取数据 赋予列名
#print(data.head())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show()

data.insert(0,'Ones',1)
cols=data.shape[1] #初始化xy
x=data.iloc[:,:-1] #x是data里的除最后列
y=data.iloc[:,cols-1:cols] #y是data最后一列
#print(x.head())
#print(y.head())

x=np.matrix(x.values)#转换x和y
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))

#print(x.shape,theta.shape,y.shape)

def computeCost(x,y,theta):
    inner = np.power(((x*theta.T)-y),2)
    return np.sum(inner)/(2*len(x))

#print(computeCost(x,y,theta)) #计算代价函数

def gradientDescent(x,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x*theta.T)-y

        for j in range(parameters):
            term=np.multiply(error,x[:,j])
            temp[0,j]=theta[0,j]-((alpha/len(x))* np.sum(term))
        theta = temp
        cost[i]=computeCost(x,y,theta)
    return theta,cost

alpha=0.01
iters=1500

g,cost=gradientDescent(x,y,theta,alpha,iters)
#print(g)

predict1 = [1,3.5]*g.T
#print("predict1:",predict1)
predict2 = [1,7]*g.T
#print("predict2:",predict2) #预测35000和70000城市规模的小吃摊利润

#x = np.linspace(data.Population.min(),data.Population.max(),100)
#f = g[0,0]+(g[0,1]*x)
'''
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Traning Date')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('predicted profit vs. population size')
#plt.show()
'''
path2 = 'D:\pytest\ex1data2.txt'
data2 = pd.read_csv(path2,header=None,names=['size','bedrooms','price'])
#print(data2.head())

data2 = (data2-data2.mean())/data2.std() #size变量是bedrooms变量的倍数  将每类特征减去平均值后除以标准差
#print(data2.head())

data2.insert(0,'Ones',1) #加一列常数项
cols = data2.shape[1] #初始化x，y
x2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

x2 = np.matrix(x2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

g2,cost2 = gradientDescent(x2,y2,theta2,alpha,iters)
#print(g2)

def normalEqn(x,y): #正规方程
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)#  x.T求矩阵的转置矩阵 .dot矩阵相乘 np.linalg.inv求矩阵的逆矩阵
    return theta
final_theta2 = normalEqn(x,y)
#print(final_theta2)





