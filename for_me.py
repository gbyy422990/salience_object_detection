import matplotlib.pyplot as plt
from skimage import data,draw,color,transform,feature,util
import cv2
import numpy as np
#加载图片，转换成灰度图并检测边缘
image_rgb = cv2.imread('20180202_123114(0).jpg',0)
sp = image_rgb.shape
for i in range(sp[0]):
    for j in range(sp[1]):
        # if image_rgb[i,j,0]==255 & image_rgb[i,j,1]==255 & image_rgb[i,j,2]==255:
        #     image_rgb[i,j,0]=image_rgb[i,j,1]=image_rgb[i,j,2]=255
        # else:
        #     image_rgb[i,j,0]=image_rgb[i,j,1]=image_rgb[i,j,2]=0
        if image_rgb[i,j]==255:
            image_rgb[i,j]=255
        else:
            image_rgb[i,j]=0
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(image_rgb,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 1)

cv2.imwrite('label.jpg',dilation)