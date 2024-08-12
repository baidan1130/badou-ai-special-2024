import cv2
import random
import numpy as np

def gaussion_noise(src,mu,sigma,percentage):
    noise_num=int(percentage*src.shape[0]*src.shape[1])
    noise_img = src
    for i in range(noise_num):
        ranX= random.randint(0,src.shape[0]-1)
        ranY= random.randint(0,src.shape[1]-1)
        noise_img[ranX,ranY]+=int(random.gauss(mu,sigma))
        if noise_img[ranX,ranY] > 255:
            noise_img[ranX, ranY] = 255
        elif noise_img[ranX,ranY] > 255:
            noise_img[ranX,ranY] = 255
    return noise_img
src=cv2.imread("lenna.png", 0)
cv2.imshow("src", src)
dst=gaussion_noise(src, 2, 6, 0.8)
cv2.imshow("dst", dst)
cv2.waitKey()
