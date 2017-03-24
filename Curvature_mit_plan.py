import h5py
import scipy
from scipy import interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def length(array,scale):
    #gibt einen array mit (nummer des pixels i,laenge des vektors von dem pixel i bis zum pixel i+scale) zurueck

    size = array.shape[0]-1
    new = np.zeros((size,2))


    for i in xrange(0,size):
        new[i,0]=scale+i
        new[i,1]=np.linalg.norm(array[i])/2+np.linalg.norm(array[i+1])/2
    #new=new[:-1]

    plt.figure()
    a1 = plt.scatter(*zip(*new), color="red")
    plt.suptitle('Length')
    return new

def length2(array):
    #gibt einen array mit (nummer des pixels i,laenge des vektors von dem pixel i bis zum pixel i+scale) zurueck
    print "jojojo: ",array
    size = array.shape[0]
    new = np.zeros((size))


    for i in xrange(0,size):

        new[i]=np.linalg.norm(array[i])
    #new=new[:-1]


    return new

def grad(curve, scale):
    size = curve.shape[0]
    new = np.array([[0, 0, 0]])
    i=0
    for i in xrange(scale-1,size-scale):
        if i >= scale:
            if (i + scale) < size+1:
                new =    np.concatenate((new, [np.subtract(curve[i + scale], curve[i - scale])]), axis=0)
    new=new[1:]

    return new




def winkel(data):
    #Winkel zwischen vektoren im array berechnen
    size=data.shape[0]-1
    array=np.zeros(size)

    for i in xrange(0,size):


        x = data[i]
        y = data[i+1]
        dot = np.dot(x, y)
        x_modulus = np.sqrt((x * x).sum())
        y_modulus = np.sqrt((y * y).sum())
        cos_angle = dot / x_modulus / y_modulus
        angle = np.arccos(cos_angle)  # Winkel in Bogenmas
        array[i]=angle

    print "array datatype: ", array.dtype
        #angle= angle * 360 / 2 / np.pi  # Winkel in Gradmas
    return array



def curvature_berechnen(data):
    #array aus winkeln und laengen erzeugen

    len1=length(grad(data,1),0)
    array_winkel=winkel(np.diff(data, axis=0))
    len=array_winkel/len1[:,1] #dphi/ds
    len1[:, 1]=len
    plt.figure()
    len2=len1
    len2[:,1]=array_winkel
    a3 = plt.scatter(*zip(*len2), color="red")
    plt.suptitle('Winkel')
    len1[:, 1]=len
    print "len1 datatype: ",len1.dtype

    return len1






def peaks_finden(curv,new,grad=0.1):
    small = np.array([(elem1, elem2) for (elem1, elem2) in curv if elem2>grad])
    orte = [elem1 for (elem1, elem2) in small]
    new_new= np.array(new).transpose()
    small_array = np.array([elem1 for elem1 in np.array(new_new)[orte]])
    where = np.where(np.diff(np.array(orte)) > 1)
    where = where[0]

    ein = []
    i0 = 0
    i1 = 0

    for item in where:
        ein.append((np.array(small_array[i0:(where[i1] + 1)])))
        i0 = where[i1] + 1
        i1 = i1 + 1

    ein.append((np.array(small_array[i0:])))


    return ein


def Curvature_zeichnen(data):
    data = np.array([(elem1, elem2, elem3 * 10) for elem1, elem2, elem3 in data])
    data = data.transpose()

    # now we get all the knots and info about the interpolated spline
    tck, u = interpolate.splprep(data, s=50000)

    # here we generate the new interpolated dataset,
    # increase the resolution by increasing the spacing, 500 in this example
    new = interpolate.splev(np.linspace(0, 1, 10000), tck)

    berechnete_curvature = curvature_berechnen(np.array(new).transpose())
    plt.figure()
    a2 = plt.scatter(*zip(*berechnete_curvature), color="red")
    plt.suptitle('Curvature')
    plt.show()
    print "curvature datatype: ", berechnete_curvature.dtype
    grad = input("Gebe den grad der Kruemmung an: ")
    # if type(grad) !='float' or grad<0:
    #    print "Falsche Eingabe, Grad wurde auf 0.1 gesetzt"
    #    grad =0.1

    stark = peaks_finden(berechnete_curvature, new, grad)
    a=int(input("Gebe a ein: "))
    b = int(input("Gebe b ein: "))
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.plot(data[0], data[1], data[2], label='original_true', lw=2, c='Dodgerblue')  # gezackt
    ax.plot(new[0], new[1], new[2], label='fit_true', lw=3, c='red')  # plot
    ax.plot(new[0][a:b], new[1][a:b], new[2][a:b], label='length', lw=4, c='green')  # plot

    # print "len(ein): ",len(ein)
    i2 = 0
   # while i2 < len(stark):
    #    ax.plot(stark[i2].transpose()[0], stark[i2].transpose()[1], stark[i2].transpose()[2], lw=4, c='yellow')
    #    i2 = i2 + 1

    ax.legend()

    plt.show()
    return stark


def printname(name):
    print name






f = h5py.File("/home/axeleik/Documents/data/cremi.splB.paths.h5", mode='r')

f.visit(printname)

data = np.array(f["z_predict1/truepaths/z/1/beta_0.5/93/0"])


Curvature_zeichnen(data)



