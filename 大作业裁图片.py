from PIL import Image
import os
import cv2
path = "D:/notfu/"
file_list = os.listdir(path)
# 循环
for image in file_list:
    img=cv2.imread(path + image)
    img0= img[0:350,0:350]		#图片裁切
    img1 = Image.fromarray(img0)
    gray = img1.convert('L')
    gray.save(path + image)
