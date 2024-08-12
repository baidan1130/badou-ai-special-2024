import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    返回值：
    retval:算法的输出状态，通常为0表示成功
    bestLabels:每个样本的最终聚类标签
    centers:最终的聚类中心
    
    data表示聚类数据，最好是np.float32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
         1.其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
         2.max_iter:最大迭代次数
         3.epsilon：精度阈值
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
"""

#读取灰度图
img=cv2.imread("lenna.png", 0)
print(img)
#获取图像高度宽度，二维转一维
h,w =img.shape[:2]
#np.reshape(a,newshape,order="C") a需要重塑的原始数组，newshape:新形状，例如（3，2）表示三行两列，若某个维度设置为-1会自动计算该维度大小
data=img.reshape((h*w,1))
print(data)
data=np.float32(data)

#停止条件
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#选择初始中心
flag=cv2.KMEANS_RANDOM_CENTERS

#Kmeans聚成四类
retval, bestLabels, centers=cv2.kmeans(data,4,None,criteria,10,flag)

#生成最终图像
dst=bestLabels.reshape(img.shape[0],img.shape[1])

#用来正常显示中文标签,防止乱码
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
title = ["原始图像","聚类图像"]
image = [img,dst]
for i in range(2):
   plt.subplot(1,2,i+1)
   """
   plt.subplot(nrows,ncols,plot_number)
   nrows 子图的行数
   ncols 子图的列数
   plot_number 要处理的子图序号，从1开始
   """
   plt.imshow(image[i], 'gray')
   plt.title(title[i])
   plt.xticks([]),plt.yticks([])     #删除图像中x/y刻度
plt.show()



