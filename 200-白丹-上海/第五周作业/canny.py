import cv2

img=cv2.imread("lenna.png", 0)
img_canny=cv2.Canny(img,100,200)
cv2.imshow("canny",img_canny)
cv2.waitKey()