

import h5py
import scipy
from scipy import interpolate
import numpy as np
import curvature

f = h5py.File("/home/axeleik/Downloads/cremi.paths.crop.split_z.h5", mode='r')
data = np.array(f["z_predict0/truepaths/z/0/beta_0.5/247/0"])[:,:]
data2 = np.array(f["z_predict0/falsepaths/z/0/beta_0.5/247/0"])[:,:]

data = np.array([(elem1, elem2,elem3*10) for elem1, elem2,elem3 in data])
#data2 = np.array([(elem1, elem2,elem3*10) for elem1, elem2,elem3 in data2])
#This is your data, but we're 'zooming' into just 5 data points
#because it'll provide a better visually illustration
#also we need to transpose to get the data in the right format
data = data.transpose()
#data2 = data2.transpose()

#now we get all the knots and info about the interpolated spline
tck, u= interpolate.splprep(data)
#tck2, u2= interpolate.splprep(data2)
#here we generate the new interpolated dataset,
#increase the resolution by increasing the spacing, 500 in this example
new = interpolate.splev(np.linspace(0,1,500), tck)
print len(new)
#new2 = interpolate.splev(np.linspace(0,1,500), tck2)

#now lets plot it!
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure(1)
ax = Axes3D(fig)
ax.plot(data[0], data[1], data[2], label='original_true', lw =2, c='Dodgerblue')
#ax.plot(data2[0], data2[1], data2[2], label='original_false', lw =2, c='Green')
ax.plot(new[0], new[1], new[2], label='fit_true', lw =3, c='red')
#ax.plot(new2[0], new2[1], new2[2], label='fit_false', lw =2, c='Yellow')
#plt.savefig('/home/axeleik/Downloads/junk.png')

x=int(input("Skala: "))
curves=curvature.plot_tf(np.array(new).transpose(),"t",x,10,20,1,2)


small = np.array([(elem1, elem2) for (elem1, elem2) in curves if elem2<6.0])
print "small: ", small
orte=[elem1 for (elem1, elem2) in small]
print "orte: ", orte
print len(orte)
new_new= np.array(new).transpose()
#print "\n\nnew_new",new_new.transpose()
print "new_new.shape:",new_new.shape

small_array = np.array([elem1 for elem1 in np.array(new_new)[orte]])
print "small_array: ", small_array
print len(small_array)
new_new=small_array.transpose()
print "new_new:",new_new
print len(new_new[1])
#ax.plot(new_new[0], new_new[1], new_new[2], label='small', lw =2, c='yellow')

ax.legend()


plt.show()

