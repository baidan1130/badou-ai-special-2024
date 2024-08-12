import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

#1.灰度化
img=cv2.imread("lenna.png", 0)

#2.高斯平滑
sigma=0.5   #高斯核参数，标准差
dim=5       #高斯核尺寸
Gauss_core=np.zeros([dim,dim])    #创建dim*dim的二维数组卷积核
tmp = [i - dim // 2 for i in range(dim)]   #高斯核各角点位置
for i in range(dim):
    for j in range(dim) :
        Gauss_core[i,j]=1/(2*math.pi*sigma**2)*math.exp(-1/(2*sigma**2)*(tmp[i]**2+tmp[j]**2)) #带入公式
h,w =img.shape[:2]
img_new = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        img_new[i,j]=np.sum(img[i:i+dim,j:j+dim]*Gauss_core)
cv2.imshow("new",img_new)
cv2.waitKey()

# 3.sobel算法（检测图像中水平、垂直和对角边缘）
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
dx,dy =img_new.shape[:2]
img_sobel_x = np.zeros([dx,dy])  # 存储梯度图像
img_sobel_y = np.zeros([dx,dy])
img_sobel = np.zeros([dx,dy])

for i in range(dx):
    for j in range(dy):
            img_sobel_x[i, j] = np.sum(img_new[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
            img_sobel_y[i, j] = np.sum(img_new[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
            img_sobel[i, j] = np.sqrt(img_sobel_x[i, j]**2 + img_sobel_y[i, j]**2)
cv2.imshow("sobel",img_sobel)
cv2.waitKey()

# 4.非极大值抑制
tan=img_sobel_y/img_sobel_x
img_yizhi = np.zeros(img_sobel.shape)
for i in range(1,dx):
    for j in range(1,dy):
        flag=True
        temp=img_sobel[i-1:i+2,j-1:j+2]
        if tan[i,j]<=-1
            num_1 = (temp[0, 1] - temp[0, 0]) / tan[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / tan[i, j] + temp[2, 1]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        elif tan[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / tan[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / tan[i, j] + temp[2, 1]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        elif tan[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * tan[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * tan[i, j] + temp[1, 0]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        elif tan[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * tan[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * tan[i, j] + temp[1, 2]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_sobel[i, j]

# 5.双阈值检测
lower_boundary = img_yizhi.mean() * 0.5      #平均值
high_boundary = lower_boundary * 3
zhan=[]
for i in range(1,img_yizhi.shape[0]) :
    for j in range(1,img_yizhi.shape[1]) :
        if img_yizhi[i, j] >= high_boundary:
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:
            img_yizhi[i, j] = 0

while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255
        zhan.append([temp_1 - 1, temp_2 - 1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

# 二值化
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

cv2.imshow("end",img_yizhi)
cv2.waitKey()