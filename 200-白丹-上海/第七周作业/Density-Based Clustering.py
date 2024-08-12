import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris=datasets.load_iris()
X=iris.data[:,:4]     #取特征空间的前四个维度

dbscan=DBSCAN(eps=0.4,min_samples=9)
dbscan.fit(X)
label=dbscan.labels_
print(label)

#绘图
x0=X[label==0]
x1=X[label==1]
x2=X[label==2]

plt.scatter(x0[:,0],x0[:,1],c="red",marker='o',label="label0")
plt.scatter(x1[:,0],x1[:,1],c="green",marker='*',label="label1")
plt.scatter(x2[:,0],x2[:,1],c="blue",marker='+',label="label2")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)  #添加图例
plt.show()