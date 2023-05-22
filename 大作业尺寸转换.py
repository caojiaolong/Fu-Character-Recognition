import os
from PIL import Image
import os.path
import glob2
def convertjpg(filename,outdir,width=500,height=500):		#尺寸转换
    img=Image.open(filename)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(filename)))
    except Exception as e:
        print(e)

for filename in glob2.glob('D:/notfu/*.png'):
    convertjpg(filename,'D:/notfu')