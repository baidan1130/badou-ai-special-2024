import cv2
import matplotlib.pyplot as plt
import numpy as np

src= cv2.imread("lenna.png")
data=src.reshape((-1,3))    #三通道
data=np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flag = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成2类
retval, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flag)

#K-Means聚类 聚集成4类
retval, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flag)

#K-Means聚类 聚集成8类
retval, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flag)

#K-Means聚类 聚集成16类
retval, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flag)

#K-Means聚类 聚集成64类
retval, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flag)

centers2 = np.uint8(centers2)
res2 = centers2[labels2.flatten()]
#labels4.flatten()将labels4变为一维数组
print(labels2.shape)
print(res2.shape)
print(centers2)
dst2=res2.reshape((src.shape))
dst2=cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)

centers4 = np.uint8(centers4)
res4 = centers4[labels4.flatten()]
dst4=res4.reshape((src.shape))
dst4=cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)

centers8 = np.uint8(centers8)
res8 = centers8[labels8.flatten()]
dst8=res8.reshape((src.shape))
dst8=cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)

centers16 = np.uint8(centers16)
res16 = centers16[labels16.flatten()]
dst16=res16.reshape((src.shape))
dst16=cv2.cvtColor(dst16,cv2.COLOR_BGR2RGB)

centers64 = np.uint8(centers64)
res64 = centers64[labels64.flatten()]
dst64=res64.reshape((src.shape))
dst64=cv2.cvtColor(dst64,cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif']=['SimHei']

titles=["原始图像","聚类图像 K=2","聚类图像 K=4","聚类图像 K=8","聚类图像 K=16","聚类图像 K=64",]
images=[src,dst2,dst4,dst8,dst16,dst64]

for i in range(6):
    plt.subplot(3,2,i+1)
    plt.title(titles[i])
    plt.imshow(images[i])
plt.show()

