import h5py
import numpy as np
import matplotlib.pyplot as plt


def grad(curve, scale):
    size = curve.shape[0]
    new = np.array([[0, 0, 0]])
    i=0
    for i in xrange(scale,size-scale):
        if i >= scale:
            if (i + scale) < size:
                new = np.concatenate((new, [np.subtract(curve[i + scale], curve[i - scale])]), axis=0)

    return new

def length(array,scale):
    size = array.shape[0]
    new = np.zeros((size,2))
    print "joj:", new
    for i in xrange(0,size-1):
        new[i,0]=scale+i
        new[i,1]=np.linalg.norm(array[i+1])

    return new


f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')

curve = np.array(f["z_predict0/truepaths/z/0/beta_0.5/11/0"])[:,:]

print curve.shape
print np.zeros(curve.shape).shape
print curve.shape[0]

gradi=grad(curve,3)
print "gradi: \n",gradi
#print "jojojo:",gradi[2, 1]
#test=np.zeros((gradi.shape[0],2))
#print test.shape
print "jojojo",gradi[1,0],gradi[1,1],gradi[1,2]
print "hi: ",np.linalg.norm(gradi[1])
len=length(gradi,3)
print len

"""new = np.array([[1,2,3]])
new2= np.array([[4,5,6]])
print np.concatenate((new , np.subtract(new2,new)),axis=0)
new = np.concatenate((new , np.subtract(new2,new)),axis=0)
print new
#print curve
#for i in curve:
 #   print "\n",i

#curvetest=np.array((curve[0],curve[1]))
#print "\n","curvetest:","\n", curvetest
#b=np.array([[1, 2, 3]])
#curvetest2=np.concatenate((curvetest,b),axis=0)
#print "\n","curvetest2:","\n", curvetest2
#print "\n","Kurve:","\n",np.array((curve[0],curve[1]))
f.close()
#print "\n","Array:","\n", np.array([[1, 2, 6], [3, 4, 5]])

#gradient= np.gradient(curve,1)
#print "\n",gradient

"""






