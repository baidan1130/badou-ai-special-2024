import cv2
import numpy as np

img=cv2.imread('lenna.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#创建sift
sift=cv2.xfeatures2d.SIFT_create()
#keypoints SIFT算法中检测到的关键点； descriptor 128维的特征向量，描述特征点的信息  None选择某片区域
keypoints,descriptor=sift.detectAndCompute(img_gray,None)

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
img=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS(image=img,outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51, 163, 236))

cv2.imshow("sift_keypoint",img)
cv2.waitKey()