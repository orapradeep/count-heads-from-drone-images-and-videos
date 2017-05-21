import matplotlib.pyplot as plt
import sys

#print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
#files_list=[]
#for i in range(1,len(sys.argv)):
#	files_list.append(sys.argv[i])
arr=[]
arr2=[]
f = open('res1.txt', 'r')
li1=f.read().split('\n')
lli1=len(li1)
li1=li1[0:lli1-1]
for idx,i in enumerate(li1):
	f = open('text/'+i+'.txt', 'r')
	arr.append(len(f.read().split('\n')) -1) # python will convert \n to os.linesep
	arr2.append(idx+1)
	f.close()  # you can omit in most cases as the destructor will call it
print arr
plt.plot(arr2,arr,'ro')
plt.plot(arr2,arr)
plt.show()

