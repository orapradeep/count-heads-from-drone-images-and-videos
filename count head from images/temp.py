import numpy as np
import cv2
from PIL import Image
import numpy as np
import math
#cap = cv2.VideoCapture('vtest.avi')
image_name='im1.png'
image_name='v.jpg'
im = Image.open(image_name).convert("RGB")
mat = np.array(im)
vmatv=mat
gray_image=8
row1=mat.shape[0]
col1=mat.shape[1]
fgbg=cv2.bgsegm.createBackgroundSubtractorMOG(noise=0.3)
#fgbg = cv2.createBackgroundSubtractorMOG()
#while(1):
#  ret, frame = cap.read()
fgmask = fgbg.apply(vmatv)
print fgmask
img = Image.fromarray(fgmask)
img=img.convert("RGB")
img.save('a.png')
img.show()	
#cv2.imshow('frame',fgmask)
#k = cv2.waitKey(30) & 0xff
#if k == 27:
#break
#cap.release()


