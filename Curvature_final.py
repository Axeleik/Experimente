import h5py
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
#from neuro_seg_plot import NeuroSegPlot as nsp
import numpy as np
#from mayavi import mlab




def length(array,scale):
    #gibt einen array mit (nummer des pixels i,laenge des vektors von dem pixel i bis zum pixel i+scale) zurueck


    size = array.shape[0]-3
    new = np.zeros((size,2))


    for i in xrange(0,size):
        new[i,0]=scale+i
        new[i,1]=(np.linalg.norm(array[1+i])+np.linalg.norm(array[2+i]))/2



    return new

def grad(curve, scale):
    size = curve.shape[0]
    new = np.array([[0, 0, 0]])
    i=0
    for i in xrange(scale-1,size-scale):
        if i >= scale:
            if (i + scale) < size+1:
                new =np.concatenate((new, [np.subtract(curve[i + scale], curve[i - scale])]), axis=0)

    new=new[1:]
    return new




def winkel(data):
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

    array_dx2 = np.zeros(size-2)

    for i in xrange(0, size-2):
        array_dx2[i]=array[i]+array[2+i]-2*array[1+i]






    array2=np.zeros((size-2,2))

    for i in xrange(0,size-2):
        array2[i,0] = 1 + i

    array2[:,1]= deepcopy(abs(array_dx2))

    return array_dx2


def maximum_ausgeben(data):

    new,data=umwandeln(data)

    maximum=np.amax(curvature_berechnen(np.array(new).transpose()), axis=0)

    return maximum[1]





def curvature_berechnen(data):

    len1 = length(np.diff(data,axis=0), 0)
    array_winkel = winkel(np.diff(data, axis=0))
    len = np.abs(array_winkel / len1[:, 1]/ len1[:, 1])# dphi/ds
    len1[:, 1] = len
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

def umwandeln(data):
    data = np.array([(elem1 * 10, elem2, elem3) for elem1, elem2, elem3 in data])
    data = data.transpose()

    tck, u = interpolate.splprep(data, s=3500,k=3)

    new = interpolate.splev(np.linspace(0, 1, 100000), tck)

    return new,data


def Curvature_zeichnen(data,data2):
    new,data = umwandeln(data)

    berechnete_curvature = curvature_berechnen(np.array(new).transpose())
    plt.figure()
    a3 = plt.scatter(*zip(*(berechnete_curvature)), color="red",linewidth='0.000001',s=10)
    plt.suptitle('curvature')
    plt.show()




    grad = input("Gebe den grad der Kruemmung an: ")

    print "data: ", data
    #data2 = [[-100, 700], [300, 900], [600, 1200]]
    stark = peaks_finden(berechnete_curvature, new, grad)

    a = int(input("gib a ein: "))
    b = int(input("gib b ein: "))


    path = np.array([[1, 1, 1], [20, 10, 10], [20, 20, 10], [20, 30, 20]], dtype=np.float64)
    path = np.swapaxes(path, 0, 1)
    #nsp.start_figure()
    #nsp.add_path(path)
    #nsp.show()
    """
    """
    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.plot(data[0], data[1], data[2], label='original_true', lw=2, c='Dodgerblue')  # gezackt
    ax.plot(data2[0], data2[1], data2[2], label='original_true', lw=2, c='Dodgerblue')  # gezackt
    ax.plot(new[0], new[1], new[2], label='fit_true', lw=1, c='red')  # plot
    ax.plot(new[0][a:b], new[1][a:b], new[2][a:b], label='Bereich', lw=5, c='green')
    print "Stark: ", stark
    i2 = 0
    if len(stark[0])!=0:
        while i2 < len(stark):
            ax.plot(stark[i2].transpose()[0], stark[i2].transpose()[1], stark[i2].transpose()[2], lw=4, c='yellow')
            i2 = i2 + 1

    ax.legend()

    plt.show()

    return berechnete_curvature




def printname(name):
    print name

if __name__ == "__main__":
    pass
    f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')





    data = np.array(f["z_train1_predict0/truepaths/z/1/beta_0.5/98/0"])

    print maximum_ausgeben(data)
    data2 = [[0, 700], [920, 1620], [280, 980]]
    print Curvature_zeichnen(data,data2)





"""
z_train1_predict0/truepaths/z/1/beta_0.5/98/0
0.003
data2 = [[0, 700], [920, 1620], [280, 980]]

z_predict0/truepaths/z/0/beta_0.5/216/0
0.003
data2 = [[0, 550], [1000, 1550], [600, 1150]]


z_train1_predict0/truepaths/z/1/beta_0.5/150/0
0.003
data2 = [[0, 700], [920, 1620], [280, 980]]

z_predict0/truepaths/z/0/beta_0.5/30/0
data2 = [[0, 700], [875, 1575], [30, 730]]

z_train1_predict0/truepaths/z/1/beta_0.5/65/0

"""
