# -*- coding: utf-8 -*-

import cv2 as cv
import os
import numpy as np
from PIL import Image
from skimage import data,filters,color
import matplotlib.pyplot as plt


from_path="/home/data1/11server/2K/Train/HR/"
data_path="/home/workplace/yz/works/EDSR-Keras-master/imgs/train/lr"
label_path="/home/workplace/yz/works/EDSR-Keras-master/imgs/train/hr"

files=os.listdir(from_path)
index=1
for file in files:
    print('正在处理图像： %s' % index)
    img_path=from_path+file
    print(img_path)
    img = cv.imread(img_path)
    hr_size = (480, 320)
    lr_size = (240, 160)
    img_hr = cv.resize(img, hr_size, interpolation = cv.INTER_CUBIC)
    img_lr = cv.resize(img_hr, lr_size, interpolation = cv.INTER_CUBIC)
    cv.imwrite(data_path+'/'+file,img_lr)
    cv.imwrite(label_path+'/'+file,img_hr)
    print('处理成功： %s' % index)
    
    index += 1
