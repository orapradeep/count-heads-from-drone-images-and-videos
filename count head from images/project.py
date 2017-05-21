from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
pp=-5
def give_border(x,imm,l1,l2):
	i1=x[0]
	j1=x[1]
	lr1=imm.shape[0]
	lc1=imm.shape[1]


	i=i1
	if(i<lr1):
		for j in range(j1,j1+l2):
			if(j<lc1):
				imm[i,j,0]=255
	i=i1+1
	if(i<lr1):
		for j in range(j1,j1+l2):
			if(j<lc1):
				imm[i,j,0]=255
	i=i1+l1-2
	if(i<lr1):
		for j in range(j1,j1+l2):
			if(j<lc1):
				imm[i,j,0]=255
	i=i1+l1-1
	if(i<lr1):
		for j in range(j1,j1+l2):
			if(j<lc1):
				imm[i,j,0]=255

	j=j1
	if(j<lc1):
		for i in range(i1,i1+l1):
			if(i<lr1):
				imm[i,j,0]=255
	j=j1+1
	if(j<lc1):
		for i in range(i1,i1+l1):
			if(i<lr1):
				imm[i,j,0]=255
	j=j1+l2-1
	if(j<lc1):
		for i in range(i1,i1+l1):
			if(i<lr1):
				imm[i,j,0]=255
	j=j1+l2-2
	if(j<lc1):
		for i in range(i1,i1+l1):
			if(i<lr1):
				imm[i,j,0]=255



	return imm

import scipy.misc

#image_name='ii1.png'
#im = Image.open(image_name).convert("RGB")
#mat = np.array(im)
#vmatv=mat
#gray_image=8
#row1=mat.shape[0]
#col1=mat.shape[1]
import sys

#print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
image_name=sys.argv[1]

stride=int(sys.argv[2])
thrv1=int(sys.argv[3])
save_image=sys.argv[4]
save_text=sys.argv[5]




im = Image.open(image_name).convert("RGB")
mat = np.array(im)
vmat1=mat
gray_image=8
row1=mat.shape[0]
col1=mat.shape[1]
#print vmat1[0:20,0:20,:]
'''
x=vmat1
for i in range(row1-2):
	for j in range(col1-2):
		sums=0
		for rna in range()
		x[i,j]=vmat1[i,j]+vmat1[i+1,j+1]+vmat1[i+2,j+2]+vmat1[i+1,j+2]+vmat1[i,j+1]+vmat1[i+1,j]+vmat1[i+1,j+2]+vmat1[i,j+2]+vmat1[i+2,j]
		x[i,j]=(x[i,j]/9.0)
vmat1=x
'''

#print vmat1[0:20,0:20,:]
print row1
print col1

mr1=280
mr2=780
mc1=540
mc2=1150
'''
img = Image.fromarray(vmat1[mr1:mr2,mc1:mc2])
img=img.convert("L")
img.save('a.png')
#img.show()	
#raw_input()
#print vmat1
#im.show()
'''
#vmat1=vmat1[mr1:mr2,mc1:mc2]

l1=vmat1.shape[0]
l2=vmat1.shape[1]
print (l1,l2)
n=80
x1=int(math.floor(l1*0.33))
x2=int(math.floor(l2*0.33))


def give_three(xv):
	l1=xv.shape[0]
	l2=xv.shape[1]
	vmat2=np.zeros((l1,l2,3))
	vmat2[:,:,0]=xv
	vmat2[:,:,1]=xv
	vmat2[:,:,2]=xv
	vmat2=vmat2.astype(np.uint8)
	return vmat2



def filter():
	image_name='im2.png'
	im = Image.open(image_name).convert("L")
	fil = np.array(im)

	#print fil
	
	tp=(fil<50)
	tp2=(fil>50)
	#print fil
	filter1=np.zeros(fil.shape)
	for i in range(fil.shape[0]):
		for j in range(fil.shape[1]):
			if(fil[i][j]<50):
				filter1[i][j]=pp
			else:
				filter1[i][j]=0	

	#print filter1			
	filter1[fil<50]=pp
	filter1[fil>=50]=0
	fils=10
	filter1=pp*np.ones((10,10,3))
	#print filter1
	#print np.sum(np.sum(filter1))
	fil=give_three(fil)
	fil[tp,0]=255
	img = Image.fromarray(fil)
	img=img.convert("RGB")
	img.save('filr.png')
#	img.show()	
	return filter1

#y=filter()
y=pp*np.ones((10,10,3))
#print y
import matplotlib.pyplot as plt

def convolve(x,y):
	sy1=y.shape[0]
	sy2=y.shape[1]
	sx1=x.shape[0]
	sx2=x.shape[1]
	#stride=5
	answers=[]
	for i in range(0,sx1-sy1,stride):
		for j in range(0,sx2-sy2,stride):
			tp=x[i:i+sy1,j:j+sy2]
#			print tp.shape
		#	print np.sum(np.sum(tp<0))
		#	print np.sum(np.sum(y>0))
		#	print np.multiply(tp,y)
		#	raw_input()


			curr=np.sum(np.sum(np.sum(np.multiply(tp,y))))
		#	print curr
			answers.append(curr)
	answers.sort()
	lans=len(answers)
	#print lans,'lans'
	an2=answers[lans-200:lans]
	#print answers[lans-1]

	#print an2
	spots=[]

	zz=np.copy(x)
	for i in range(0,sx1-sy1,stride):
		for j in range(0,sx2-sy2,stride):
			tp=zz[i:i+sy1,j:j+sy2]
			curr=np.sum(np.sum(np.multiply(tp,y)))
			if(curr in an2):
				spots.append([i,j])
				zz[i:i+sy1,j:j+sy2]=255

	return spots

def convolve2(x,y):
	sy1=y.shape[0]
	sy2=y.shape[1]
	sx1=x.shape[0]
	sx2=x.shape[1]
	stride=5
	answers=[]
	thr=np.sum(np.sum(np.sum(np.multiply(y,y))))
	thr=thr*1.13
	#print thr
	extra=x

	for i in range(0,sx1-sy1,stride):
		for j in range(0,sx2-sy2,stride):
			tp=x[i:i+sy1,j:j+sy2]
		#	print np.sum(np.sum(tp<0))
		#	print np.sum(np.sum(y>0))
		#	print np.multiply(tp,y)
		#	raw_input()



			curr=np.sum(np.sum(np.sum(np.multiply(tp,y))))
#			print curr
			if(curr>=thr):
				extra[i:i+sy1,j:j+sy2]=255

		#	print curr
			answers.append(curr)
	answers.sort()
	lans=len(answers)
	#print lans,'lans'
	an2=answers[lans-1000:lans]
	#print answers[lans-1]

	#print an2
	spots=[]

	zz=np.copy(x)
	for i in range(0,sx1-sy1,stride):
		for j in range(0,sx2-sy2,stride):
			tp=zz[i:i+sy1,j:j+sy2]
			curr=np.sum(np.sum(np.multiply(tp,y)))
			if(curr in an2):
				spots.append([i,j])
				zz[i:i+sy1,j:j+sy2]=255
	img = Image.fromarray(extra)
	img=img.convert("RGB")
	img.save('a.png')
#	img.show()	

	return spots
#convolve2(vmat1,vmatv)

spots=convolve(vmat1,y)
#vmat2=give_three(vmat1)

#give_border([100,100], vmat2,50,50)

thrv=thrv1
def mergeit(spots):
	flags=[]
	for idx,i in enumerate(spots):
		flags.append(0)
	for idx,i in enumerate(spots):
		if(flags[idx]==0):
			x1=(i[0]+thrv,i[1])
			x2=(i[0]-thrv,i[1])
			x3=(i[0],i[1]+thrv)
			x4=(i[0],i[1]-thrv)
			for jdx,j in enumerate(spots):
				if(jdx == idx):
					continue
				else:
					if(flags[jdx]==0):
						if(j[0]<x1[0] and j[0]>x2[0]):
							if(j[1]<x3[1] and j[1]>x4[1]):
								flags[jdx]=1
								flags[idx]=2
								#print 'hi',idx,i,j,x1,x2,x3,x4
								#print flags
	diff2=[]
	for idx,i in enumerate(spots):
		if(flags[idx]==2 ):
			diff2.append(i)
	return diff2
	return spots[0:5]

spots=mergeit(spots)
vmat2=vmat1
for i in spots:
	vmat2=give_border(i,vmat2,y.shape[0]+thrv,y.shape[1]+thrv)

f = open(save_text, 'w')
for i in spots:
	f.write(str(i)+'\n')  # python will convert \n to os.linesep
f.close()  # you can omit in most cases as the destructor will call it

n=4
vmat3=np.zeros((n,n))
lr1=float(vmat2.shape[0])
lc1=float(vmat2.shape[1])
for idx,i in enumerate(spots):
	portx=i[0]/lr1
	porty=i[1]/lc1
	portx=int(math.floor(portx*n))
	porty=int(math.floor(porty*n))
	vmat3[portx,porty]=vmat3[portx,porty]+1
vmat4=vmat3
print vmat4
vmat3=np.zeros((n*100,n*100,3)).astype('uint8')

tocol=int(np.max(np.max(vmat4)))

tocol=tocol+1
tocol=6
print tocol,'tocol'
colora=np.zeros((tocol,1  ))
for i in range(tocol):
	colora[i]=int(math.floor(((i*255.0)/tocol)))

for i in range(n):
	for j in range(n):
		n1=i*100+100
		n2=j*100+100
		tt1=int(vmat4[i][j])
		if(int(vmat4[i][j]) >= tocol):
			tt1=tocol-1
		print int(colora[tt1])
		vmat3[i*100:n1 , j*100:n2 ,0 ]=int(colora[tt1])
#print np.sum(np.sum(np.sum(vmat3>255)))
img = Image.fromarray(vmat3.astype('uint8'))
img=img.convert("RGB")
img.save('heat_map/'+save_image)
print 'hi'
img.show()	
ex1=np.ones((n,1))
img = Image.fromarray(vmat2)
img=img.convert("RGB")
img.save(save_image)
img.show()	

