import numpy as np

class PCA():
    def __init__(self,x,k):
        self.x=x                       #样本矩阵
        self.k=k                       #降维矩阵维度
        self.centerX=self.center()     #样本矩阵中心化
        self.C=self.cov()              #样本的协方差矩阵
        self.U=self.U()                #样本的降维转换矩阵
        self.Z=self.Z()                #样本的降维矩阵

    def center(self):
        mean=np.mean(self.x,axis=0)    #求每列的均值
        centerX=self.x-mean
        return centerX

    def cov(self):
        num=np.shape(self.centerX)[0]   #样本列数
        C=np.dot(self.centerX.T,self.centerX)/num
        return C

    def U(self):
        a, b = np.linalg.eig(self.C)  # 特征值赋值给a，对应特征向量赋值给b。
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        # U = eig_vectors[:,idx[:self.n_components]]
        UT = [b[:, ind[i]] for i in range(self.k)]
        U = np.transpose(UT)
        return U

    def Z(self):
        Z=np.dot(self.x,self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z



if __name__=='__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    pca_X=PCA(X,K)
    new_X=PCA.Z(pca_X)
    print(new_X)