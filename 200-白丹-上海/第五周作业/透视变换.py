import cv2
import numpy as np

img=cv2.imread("photo1.jpg")


src = np.array([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.array([[0, 0], [337, 0], [0, 488], [337, 488]])
#透视变换矩阵
m=cv2.getPerspectiveTransform(src,dst)
img_new=cv2.warpPerspective(img,m,(337,488))
cv2.imshow("new",img_new)
cv2.waitKey()

