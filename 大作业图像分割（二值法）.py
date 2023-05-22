import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
path= "D:/fu/fu (190).png"
img0=cv2.imread(path,cv2.IMREAD_GRAYSCALE)

ret0, img1 = cv2.threshold(img0, 70, 255, cv2.THRESH_BINARY_INV) #图像分割
ret1, img2 = cv2.threshold(img0, 100, 255,cv2.THRESH_BINARY_INV) #黑白转换
img3 = Image.fromarray(img2)
gray = img3.convert('L')
gray.save(path)