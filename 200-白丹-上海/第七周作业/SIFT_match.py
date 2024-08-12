import cv2
import numpy as np
# 读图
img1_gray=cv2.imread("iphone1.png", 0)
img2_gray=cv2.imread("iphone2.png", 0)

# sift算法获取特征点
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1=sift.detectAndCompute(img1_gray, None)
kp2,des2=sift.detectAndCompute(img2_gray, None)

# 特征点匹配
bf=cv2.BFMatcher(cv2.NORM_L2)  #cv2.NORM_L2距离度量方式，欧几里得距离
matches=bf.knnMatch(kp1,kp2,k=2)   #knnmatch将点全部遍历蛮力匹配，找出相似度最高的前k个
goodMatch=[:20]
for m,n in matches:
    if m.distance < 0.50*n.distance
        goodMatch.append(m)

# 绘图,两张图纸并列，特征点连线
h1,w1=img1_gray.shape[:2]
h2,w2=img2_gray.shape[:2]
    #拼接画布
vis=np.zeros((max(h1,h2),w1+w2,3),np.uint8)
vis[:h1,:w1]=img1_gray
vis[:h2,w1:w2]=img2_gray
    #查询图像与训练图像中的匹配点连线
P1=[kpp.queryIdx for kpp in goodMatch]  #查询图像特征点索引集合
P2=[kpp.trainIdx for kpp in goodMatch]  #训练图像特征点索引集合

#kp1[pp].pt特征点的坐标，将索引转化为具体坐标
#np.int32()输入数据可以是Python数字、列表、Numpy数组等
post1=np.int32([kp1[pp].pt for pp in P1])
post2=np.int32([kp2[pp].pt for pp in P2])+(w1.0)   #匹配点坐标右移w1

for (x1,y1),(x2,y2) in zip(post1,post2):
    cv2.line(vis,(x1,y1),(x2,y2),(0,0,255))
cv2.namedWindow("match",cv2.WINDOW_NORMAL)
cv2.waitKey()