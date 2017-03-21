import h5py
import scipy
from scipy import interpolate
import numpy as np
import curvature
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        #angle= angle * 360 / 2 / np.pi  # Winkel in Gradmas
    return array


def fertig(data):
    #array aus winkeln und laengen erzeugen

    len1=curvature.length(curvature.grad(data,1),0)
    array_winkel=winkel(np.diff(data, axis=0))
    len=array_winkel/len1[:,1] #dphi/ds
    len1[:, 1]=len

    plt.figure()
    a1=plt.scatter(*zip(*len1), color="red")
    plt.show()



    #print "shape diff, len1,curvature.grad(data,1),data, array_winkel:", np.diff(data, axis=0).shape,len1.shape,curvature.grad(data,1).shape,data.shape,array_winkel.shape




f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')
data = np.array(f["z_predict0/truepaths/z/0/beta_0.5/247/0"])[:, :]

data = np.array([(elem1, elem2, elem3 * 10) for elem1, elem2, elem3 in data])
data = data.transpose()


#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data,s=50000)

#here we generate the new interpolated dataset,
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,10000), tck)
fertig(np.array(new).transpose())










"""
fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot(data[0], data[1], data[2], label='original_true', lw =2, c='Dodgerblue') #gezackt
ax.plot(new[0], new[1], new[2], label='fit_true', lw =3, c='red') #plot








plt.show()

"""