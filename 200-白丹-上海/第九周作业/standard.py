import numpy as np

def normalization1(x):
    #0-1归一化
    return[(float(i)-min(x)/max(x)-min(x)) for i in x]

def normalization2(x):
    #-1-1归一化
    return [(float(i)-np.mean(x)/max(x)-min(x)) for i in x]

def z_score(x):
    sigma=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-np.mean(x))/sigma for i in x]

x=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

n1=normalization1(x)
n2=normalization1(x)
z=z_score(x)
print(n1)
print(n1)
print(z)

