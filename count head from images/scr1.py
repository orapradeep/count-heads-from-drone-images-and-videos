from subprocess import call
import shutil
import os
shutil.rmtree('images')
shutil.rmtree('output')
shutil.rmtree('text')
shutil.rmtree('heat_map')
os.makedirs('images')
os.makedirs('output')
os.makedirs('heat_map')
os.makedirs('text')

#call(['python','project.py','de1.png','5','80','sde1.png','text/tde1.txt'])
#call(['python','project.py','de2.png','5','80','sde2.png','text/tde2.txt'])
#call(['python','project.py','ii1.png','5','120','sii1.png','text/tii1.txt'])
#call(['python','project.py','ii2.png','5','120','sii2.png','text/tii2.txt'])


#call(['python','timevsp.py','text/tde1.txt','text/tde2.txt','text/tii1.txt','text/tii2.txt'])


import pyscreenshot as ImageGrab
from PIL import ImageEnhance

import time

count = 0;

f = open('res1.txt', 'w')

images=[]
count=0
time.sleep(5)
while count<5:    
    snapshot = ImageGrab.grab()
#    snapshot = ImageEnhance.Contrast(snapshot)
#    snapshot = snapshot.enhance(2)
    #snapshot = snapshot.rotate(90);
    #save_path = "C:\\Users\\Maulik\\Desktop\\Image\\MySnapshot"+str(count)+".jpg"
    save_path = str(count)+".jpg"
    snapshot.save(save_path)
    time.sleep(5)
    print 'hi1'
    call(['python','project.py','images'+save_path,'5','120','output/'+'s'+save_path,'t'+'text'+save_path+'.txt'])	
    print 'hi2'
    i1='t'+str(count)+'.jpg'
    images.append(i1)
    snapshot.save(save_path)
    count = count + 1
    time.sleep(2)
#    raw_input()

for i in images:
    f.write(str(i)+'\n')  # python will convert \n to os.linesep
f.close()  # you can omit in most cases as the destructor will call it

call(['python','timevsp.py'])









