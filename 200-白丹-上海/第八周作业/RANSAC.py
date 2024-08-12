import scipy as sp
import numpy as np
import scipy.linalg as sl


def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    '''
    :param data: 样本数据
    :param model: 确定的模型
    :param n: 生成模型所需的最小样本点
    :param k: 最大迭代次数
    :param t: 阈值：作为判断点满足模型的条件
    :param d: 拟合较好时需要的样本点最少个数
    :return:
        bestfit:最优拟合解（未找到返回ValueError）
    '''

    iterations=0
    bestfit=None
    besterr=np.inf  #无限大
    best_inlier_idxs=None
    while iterations<k:
        maybe_idxs,test_idxs=random_partition(n,data.shape[0])   #random_partition函数将数据集data(data.shape[0]为总样本数量)随机分为两部分：前n个为内点集合(maybe_idxs)，剩余为测试集（test_idxs）
        maybe_inliers=data[maybe_idxs,:]     #内群点的坐标
        test_point=data[test_idxs]           #数据点（Xi,Yi）
        maybe_model=model.fit(maybe_inliers)  #拟合模型
        test_err=model.get_error(test_point,maybe_model)   #计算平方和误差
        also_idxs=test_idxs[test_err<t]
        also_inliers=data[also_idxs,:]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print(f'iteration={iterations},len(also_inliers)={also_inliers}')

        if (len(also_inliers) >d):       #加括号的重要性：保证先进行len()，再做大小比较
        #有了较好的模型，测试模型符合度
            betterdata = np.concatenate((maybe_inliers,also_inliers))     #样本连接
            bettermodel =model.fit(betterdata)
            better_err=model.get_error(betterdata,bettermodel)
            thiserr=np.mean(better_err)
            if thiserr<besterr:          #若本次误差更小，将本次更新为bestfit
                bestfit=bettermodel
                besterr=thiserr
                best_inlier_idxs=np.concatenate((maybe_idxs,also_idxs))   #更新内群点
        iterations+=1     #设置循环结束的条件

    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit
def random_partition(n,n_data):
    all_idxs =np.arange(n_data)   #生成从0到n_data的数列，获取n_data下标索引
    np.random.shuffle(all_idxs)   #打乱下标索引
    idxs1=all_idxs[:n]
    idxs2=all_idxs[n:]
    return idxs1,idxs2

class LinearLeastSquareModel:
    #最小二乘法计算RANSAC输入模型
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns=input_columns
        self.output_columns=output_columns
        self.debug=debug

    def fit(self,data):
        A=np.vstack([data[:,i] for i in self.input_columns]).T     #np.vstack()将数组竖向堆叠，根据input_columns索引获得第一列再转置
        B=np.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s=sl.lstsq(A,B)
        #np中求最小二乘法接口   sl.lstsq(a,b)
        #输入:a 一个二维数组，表示自变量矩阵；b 一个一维或二维数组表示因变量矩阵
        #输出：x:最小二乘解，即模型的参数向量  resids:残差平方和   rank:输入矩阵的秩   s:奇异值
        return x #返回最小平方和向量

    def get_error(self,data,model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit=np.dot(A,model)  #矩阵相乘，计算y值，B_fit=model.k*A+model.b
        err_per_point=np.sum((B-B_fit)**2,axis=1) #计算每行的数据和,得到一个一维数组
        return err_per_point

def test():
    #生成理想数据
    n_samples=500 #样本个数
    n_inputs=1    #输入变量个数
    n_outputs=1   #输出变量个数
    A_exact=20*np.random.random((n_samples,n_inputs))   #n_samples行，n_inputs列，0-20的随机数(500行1列)
    perfect_fit=60*np.random.normal(size = (n_inputs, n_outputs) )  #np.random.normal(均值为0，标准差为1，尺寸size),生成服从正态分布随机数，随机生成一个斜率
    B_exact=np.dot(A_exact,perfect_fit)   #y=kx

    #加入高斯噪声
    A_noise=A_exact+np.random.normal(size=A_exact.shape)   #Xi
    B_noise=B_exact+np.random.normal(size=B_exact.shape)   #Yi

    if 1:     #布尔值True可以由任何非零值表示。if 1：始终为True,if内代码始终执行
        #添加局外点
        n_outliers=100
        all_idxs=np.arange(A_exact.shape[0])    #获取0-499索引
        np.random.shuffle(all_idxs)             #打乱
        outliers_idxs=all_idxs[:n_outliers]
        A_noise[outliers_idxs]=20*np.random.random((n_outliers,n_inputs))   #加入局外点的Xi
        B_noise[outliers_idxs]=50*np.random.normal(size=(n_outliers,n_outputs))   #加入局外点的Yi

    #建立model
    all_data=np.hstack((A_noise,B_noise))      #形成（[Xi,Yi]....）shape=500行2列
    input_columns=range(n_inputs)              #数组第一列的索引
    output_columns=[n_inputs+i for i in range(n_outputs)]      #数组最后一列的索引
    debug=False
    model=LinearLeastSquareModel(input_columns,output_columns,debug=debug)

    linear_fit,resids,rank,s=sp.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])  #最小二乘法接口

    #运行ransac算法
    ransac_fit,ransac_data=ransac(all_data,model,50,1000,7e3,300,debug=debug,return_all=True)     #7e3:7*10的三次方即7000

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 对A_exact里的数据重排

        if 1:
            pylab.plot(A_noise[:, 0], B_noise[:, 0], 'k.', label='data')  # 散点图   k.表示黑色原点
            # A_noise[ransac_data['inliers'], 0]表示ransac_data['inliers']作为索引内群点的第一列 #bx表示蓝色x
            pylab.plot(A_noise[ransac_data['inliers'], 0], B_noise[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")

        else:
            pylab.plot(A_noise[non_outlier_idxs, 0], B_noise[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noise[outlier_idxs, 0], B_noise[outlier_idxs, 0], 'r.', label='outlier data')     #r.红色圆点

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')                                       #ransac算法得到的拟合线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')                                     #自己设置的拟合线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')                                       #最小二乘法得到的拟合线
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()