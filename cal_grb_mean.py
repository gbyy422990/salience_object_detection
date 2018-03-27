#coding:utf-8
#Bin GAO

import os
import numpy as np
import cv2


image_path = '/Volumes/image'

file_names = os.listdir(image_path)

count = 0
mean = np.zeros(3, np.int64)


for i in file_names[1:]:
    print(i)

    img = cv2.imread(image_path + '/' + i)  #imread读进来的是BGR格式
    #print(img)
    count += 1
    mean += np.sum(img, axis=(0, 1)).astype(int)
h, w = img.shape[0:-1]
print(h, w, count)
means = mean / (1.0 * h * w * count)
print('b, g, r = ', means)
