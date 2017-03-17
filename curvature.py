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

    for i in xrange(0,size-1):
        new[i,0]=scale+i
        new[i,1]=np.linalg.norm(array[i+1])

    return new

def printname(name):
     print name


def plot_tf(curve,ok,scale1,scale2,scale3,a,b):
    path="Truepath"



    gradi1 = grad(curve, scale1)
   # gradi2 = grad(curve, scale2)
   # gradi3 = grad(curve, scale3)
    len1 = length(gradi1, scale1)
    #len2 = length(gradi2, scale2)
    #len3 = length(gradi3, scale3)
    len1 = np.delete(len1, curve.shape[0] - 2 * scale1, 0)
    #print "\n\ntest1:: \n\n",len1
    #len2 = np.delete(len2, curve.shape[0] - 2 * scale2, 0)
    #len3 = np.delete(len3, curve.shape[0] - 2 * scale3, 0)
    zip(*len1)
    #zip(*len2)
    #zip(*len3)
    plt.figure()
    plt.subplot(2,1,a)
    if ok in ('t', 'T'):
        plt.title('Truepath(Nicht normiert)')
    elif ok in ('f', 'F'):
        plt.title('Falsepath(Nicht normiert)')
    a1=plt.scatter(*zip(*len1), color="red")
    #a2=plt.scatter(*zip(*len2), color="blue")
    #a3=plt.scatter(*zip(*len3), color="green")
    #plt.legend((a1,a2,a3),(scale1, scale2, scale3),scatterpoints=1,loc='lower left',ncol=3,fontsize=12)

    len1 = [(elem1, elem2 / scale1) for elem1, elem2 in len1]
    #len2 = [(elem1, elem2 / scale2) for elem1, elem2 in len2]
    #len3 = [(elem1, elem2 / scale3) for elem1, elem2 in len3]


    zip(*len1)
    #zip(*len2)
    #zip(*len3)

    plt.subplot(2,1,b)
    if ok in ('t', 'T'):
        plt.title('Truepath(Normiert)')
    elif ok in ('f', 'F'):
        plt.title('Falsepath(Normiert)')
    b1=plt.scatter(*zip(*len1), color="red")
    #b2=plt.scatter(*zip(*len2), color="blue")
    #b3=plt.scatter(*zip(*len3), color="green")
    #plt.legend((b1, b2, b3), (scale1, scale2, scale3) ,scatterpoints=1,loc='lower left',ncol=3,fontsize=12)
    return len1

#f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')
#curve1 = np.array(f["z_predict0/truepaths/z/0/beta_0.5/11/0"])[:,:]
"""

scale1 = int(input("please enter scale 1: "))
scale2 = int(input("please enter scale 2: "))
scale3 = int(input("please enter scale 3: "))

curve1 = np.array(f["z_predict0/truepaths/z/0/beta_0.5/11/0"])[:,:]
curve2 = np.array(f["z_predict0/falsepaths/z/0/beta_0.5/247/0"])[:,:]
curve3 = np.array(f["z_predict0/falsepaths/z/0/beta_0.5/247/0"])[:,:]
curve4 = np.array(f["z_predict0/falsepaths/z/0/beta_0.5/259/0"])[:,:]
curve5 = np.array(f["z_predict0/falsepaths/z/0/beta_0.5/260/0"])[:,:]
f.close()
plt.figure()

plot_tf(curve1,"t",scale1,scale2,scale3,1,2)
plot_tf(curve2,"f",scale1,scale2,scale3,3,4)
plot_tf(curve3,"t",scale1,scale2,scale3,5,6)
plot_tf(curve4,"f",scale1,scale2,scale3,7,8)
plot_tf(curve5,"f",scale1,scale2,scale3,9,10)

plt.show()



f.visit(printname)
"""


