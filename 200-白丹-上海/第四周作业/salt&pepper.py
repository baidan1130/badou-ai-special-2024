import cv2
import random
import numpy as np

def SaltandPepper (src,percentage):
    noise_num=int(percentage*src.shape[0]*src.shape[1])
    noise_img= src
    for i in range(noise_num):
        ranX= random.randint(0,src.shape[0]-1)
        ranY= random.randint(0,src.shape[1]-1)
        if random.random() > 0.5:
            noise_img[ranX, ranY] = 255
        else :
            noise_img[ranX,ranY] = 0
    return noise_img
src=cv2.imread("lenna.png", 0)
cv2.imshow("src", src)
dst=SaltandPepper(src, 0.01)
cv2.imshow("dst", dst)
cv2.waitKey()