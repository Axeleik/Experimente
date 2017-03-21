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


def curvature_berechnen(data):
    #array aus winkeln und laengen erzeugen

    len1=curvature.length(curvature.grad(data,1),0)
    array_winkel=winkel(np.diff(data, axis=0))
    len=array_winkel/len1[:,1] #dphi/ds
    len1[:, 1]=len

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















f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')
data = np.array(f["z_predict0/truepaths/z/0/beta_0.5/247/0"])[:, :]

data = np.array([(elem1, elem2, elem3 * 10) for elem1, elem2, elem3 in data])
data = data.transpose()


#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data,s=50000)

#here we generate the new interpolated dataset,
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,10000), tck)
berechnete_curvature=curvature_berechnen(np.array(new).transpose())
grad=float(input("Gebe den grad der Kruemmung an: "))
stark=peaks_finden(berechnete_curvature,new,grad)




fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot(data[0], data[1], data[2], label='original_true', lw =2, c='Dodgerblue') #gezackt
ax.plot(new[0], new[1], new[2], label='fit_true', lw =3, c='red') #plot


#print "len(ein): ",len(ein)
i2=0
while i2<len(stark):
    ax.plot(stark[i2].transpose()[0], stark[i2].transpose()[1], stark[i2].transpose()[2], lw=4, c='yellow')
    i2=i2+1

ax.legend()


plt.show()

#plt.show()



