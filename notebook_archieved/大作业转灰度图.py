from PIL import Image
import os
path = './' 
file_list = os.listdir(path) 
# 循环
for image in file_list:
    if ".png" in image:
        I = Image.open(path + image)
        gray = I.convert('L')	#转为灰度图
        gray.save(path + image)	#存储