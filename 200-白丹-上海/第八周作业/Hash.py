import cv2
import numpy as np

#均值哈希
def aHash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sum = 0
    str = ""
    for i in range(8):
        for j in range(8):
            sum+=img_gray[i,j]
            avg=sum/8
            if img_gray[i,j]>avg:
                str+="1"
            else:
                str+="0"
    return str

#差值哈希
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    str = ""
    for i in range(8):
        for j in range(8):
            if img_gray[i,j]>img_gray[i,j+1]:
                str+="1"
            else:
                str+="0"
    return str

#两幅图哈希值对比

def cmpHash(hash1,hash2):
    n = 0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n+=1
    return n

img=cv2.imread("lenna.png")
img_noise=cv2.blur(img,(15,1))
ahash1=aHash(img)
ahash2=aHash(img_noise)
n=cmpHash(ahash1,ahash2)
print(f"原图的均值哈希值是{ahash1},噪声图的均值哈希值是{ahash2}，均值哈希相似度是{n}")

dhash1=dHash(img)
dhash2=dHash(img_noise)
m=cmpHash(dhash1,dhash2)
print(f"原图的差值哈希值是{dhash1},噪声图的差值哈希值是{dhash2}，差值哈希相似度是{m}")